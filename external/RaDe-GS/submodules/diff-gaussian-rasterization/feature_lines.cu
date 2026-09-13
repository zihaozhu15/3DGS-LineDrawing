// Inference-only feature-line diagnostics. Reuses RaDe-GS internal sorted splats.
#include "feature_lines.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include "cuda_rasterizer/config.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void topkKernel(const uint2* ranges, const uint32_t* list,
    const float2* means, const float4* conics, int H, int W, int K,
    int* ids, float* weights, float* opacity) {
    int x=blockIdx.x*BLOCK_X+threadIdx.x, y=blockIdx.y*BLOCK_Y+threadIdx.y;
    if(x>=W || y>=H) return;
    int p=y*W+x;
    uint2 range=ranges[blockIdx.y*gridDim.x+blockIdx.x];
    float best[8]={0}; int bid[8];
    for(int j=0;j<8;j++) bid[j]=-1;
    float T=1.f;
    for(uint32_t i=range.x;i<range.y;i++) {
        int id=list[i]; float2 xy=means[id]; float4 c=conics[id];
        float dx=xy.x-x, dy=xy.y-y;
        float power=-0.5f*(c.x*dx*dx+c.z*dy*dy)-c.y*dx*dy;
        if(power>0.f) continue;
        float a=fminf(.99f,c.w*expf(power));
        if(a<1.f/255.f) continue;
        float next=T*(1.f-a);
        if(next<.0001f) break; // Same early-termination semantics as render_forward.cu.
        float weight=a*T;
        for(int j=0;j<K;j++) if(weight>best[j]) {
            for(int l=K-1;l>j;l--) {best[l]=best[l-1];bid[l]=bid[l-1];}
            best[j]=weight;bid[j]=id;break;
        }
        T=next;
    }
    for(int j=0;j<K;j++) {ids[p*K+j]=bid[j];weights[p*K+j]=best[j];}
    opacity[p]=1.f-T;
}

__global__ void fieldsKernel(const float* rgb,const float* alpha,const float* depth,
    const float* normal,const int* ids,const float* weights,int H,int W,int K,
    float foreground,float* out) {
    int p=blockIdx.x*blockDim.x+threadIdx.x, N=H*W;
    if(p>=N) return;
    int x=p%W,y=p/W;
    float result[5]={0};
    int neighbors[4]={x>0?p-1:-1,x<W-1?p+1:-1,y>0?p-W:-1,y<H-1?p+W:-1};
    for(int n=0;n<4;n++) {
        int q=neighbors[n];if(q<0)continue;
        result[0]=fmaxf(result[0],fabsf(alpha[p]-alpha[q]));
        if(alpha[p]<foreground || alpha[q]<foreground)continue;
        float dp=depth[p],dq=depth[q];
        if(dp>0 && dq>0) result[1]=fmaxf(result[1],fabsf(dp-dq)/fmaxf(fmaxf(dp,dq),1e-6f));
        float dot=0,lp=0,lq=0,lum=0;
        const float luma[3]={.2126f,.7152f,.0722f};
        for(int c=0;c<3;c++) {
            float a=normal[c*N+p],b=normal[c*N+q];dot+=a*b;lp+=a*a;lq+=b*b;
            lum+=luma[c]*(rgb[c*N+p]-rgb[c*N+q]);
        }
        if(lp>1e-8f && lq>1e-8f) result[2]=fmaxf(result[2],1.f-fminf(1.f,fmaxf(-1.f,dot*rsqrtf(lp*lq))));
        result[3]=fmaxf(result[3],fabsf(lum));
        float sp=0,sq=0,overlap=0;
        for(int j=0;j<K;j++){sp+=weights[p*K+j];sq+=weights[q*K+j];}
        if(sp>1e-8f && sq>1e-8f) {
            for(int j=0;j<K;j++) if(ids[p*K+j]>=0)
                for(int l=0;l<K;l++) if(ids[p*K+j]==ids[q*K+l])
                    overlap+=fminf(weights[p*K+j]/sp,weights[q*K+l]/sq);
            result[4]=fmaxf(result[4],fmaxf(0.f,1.f-overlap));
        }
    }
    for(int i=0;i<5;i++)out[i*N+p]=result[i];
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> ExtractTopKCUDA(
    torch::Tensor geom,torch::Tensor binning,torch::Tensor image,
    int points,int rendered,int height,int width,int k) {
    TORCH_CHECK(k>=1 && k<=8 && height>0 && width>0,"Invalid Top-k dimensions");
    TORCH_CHECK(geom.is_cuda() && binning.is_cuda() && image.is_cuda(),"Expected CUDA buffers");
    c10::cuda::CUDAGuard guard(geom.device());
    auto ids=torch::full({height,width,k},-1,geom.options().dtype(torch::kInt32));
    auto weights=torch::zeros({height,width,k},geom.options().dtype(torch::kFloat32));
    auto alpha=torch::zeros({1,height,width},weights.options());
    if(points && rendered) {
        char* g=(char*)geom.data_ptr();char* b=(char*)binning.data_ptr();char* im=(char*)image.data_ptr();
        auto gs=CudaRasterizer::GeometryState::fromChunk(g,points);
        auto bs=CudaRasterizer::BinningState::fromChunk(b,rendered);
        auto is=CudaRasterizer::ImageState::fromChunk(im,height*width);
        dim3 grid((width+BLOCK_X-1)/BLOCK_X,(height+BLOCK_Y-1)/BLOCK_Y),block(BLOCK_X,BLOCK_Y);
        topkKernel<<<grid,block,0,at::cuda::getCurrentCUDAStream()>>>(is.ranges,bs.point_list,gs.means2D,gs.conic_opacity,
            height,width,k,ids.data_ptr<int>(),weights.data_ptr<float>(),alpha.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {ids,weights,alpha};
}

torch::Tensor FeatureFieldsCUDA(torch::Tensor rgb,torch::Tensor alpha,torch::Tensor depth,
    torch::Tensor normal,torch::Tensor ids,torch::Tensor weights,float foreground) {
    TORCH_CHECK(rgb.is_cuda() && rgb.is_contiguous() && rgb.scalar_type()==torch::kFloat32,"Expected contiguous CUDA float RGB");
    c10::cuda::CUDAGuard guard(rgb.device());
    int H=rgb.size(1),W=rgb.size(2),K=ids.size(2);
    TORCH_CHECK(rgb.size(0)==3 && K>=1 && K<=8,"Invalid field dimensions");
    for(auto t:{alpha,depth,normal,weights})
        TORCH_CHECK(t.device()==rgb.device() && t.is_contiguous() && t.scalar_type()==torch::kFloat32,"Invalid float field");
    TORCH_CHECK(alpha.numel()==H*W && depth.numel()==H*W && normal.numel()==3*H*W,"Invalid image shapes");
    TORCH_CHECK(ids.device()==rgb.device() && ids.is_contiguous() && ids.scalar_type()==torch::kInt32 && ids.numel()==H*W*K && weights.numel()==H*W*K,"Invalid Top-k fields");
    auto out=torch::empty({5,H,W},rgb.options());
    fieldsKernel<<<(H*W+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(rgb.data_ptr<float>(),alpha.data_ptr<float>(),
        depth.data_ptr<float>(),normal.data_ptr<float>(),ids.data_ptr<int>(),weights.data_ptr<float>(),H,W,K,foreground,out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
