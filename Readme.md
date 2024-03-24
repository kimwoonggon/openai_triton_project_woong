Openai triton project for visual transformer implementation  
Dockerfile Build  
```
docker build -f tritonProject.Dockerfile -t tritonproject:cuda121gogo .
```
Docker container start
```
docker run -it --rm --gpus device=0 --ulimit memlock=-1 --ulimit stack=-1 --ulimit core=-1 --ipc=host --shm-size=32gb --name tritonproject -v $(pwd):/mnt tritonproject:cuda121gogo
```
Environments  
ubnutu 20.04  
GPU: A100  
torch==2.1.0a0+32f93b1  
cuda==12.2  
cudnn==8.9.0.5  

