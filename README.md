[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-0.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-1.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-2.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-3.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-4.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-5.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-6.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-7.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-8.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-9.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-10.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-11.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-12.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-13.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-14.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-15.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-16.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-17.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-18.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-19.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-20.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-21.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-22.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)
[![TFDM](/doc/Timestep_Free_Diffusion_Model-images-23.jpg)](/doc/Timestep_Free_Diffusion_Model.pdf)

# todo:

1. iDDPM:cosine_beta_schedule
2. dataset: CIFAR10 or ImageNet
3. x,t = TCDM(x,TFDM(x))
4. [x => down(x) => x,t => up(x,t) => x ]
4. t_loss: L1 --> cross_entropy ?
5. TFDM is working as soon as t can be predicted from x_t on the bigger dataset
6. torch.randn(6, hidden_size) / hidden_size**0.5 # need scaling for the initialization
7. 之所以在middle失败，是因为如图，在linear Schedule中，靠近0.5的地方，才是真正去噪的起点。
[![TFDM](/doc/Snipaste_2024-08-23_12-35-40.png)](linear_schedule vs cosine_schedule)