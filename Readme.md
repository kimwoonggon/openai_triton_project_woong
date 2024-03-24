### 도커 파일을 활용한 빌드 
```
docker build -f tritonProject.Dockerfile -t tritonproject:cuda121gogo .
```
### 도커 컨테이너 실행하기
```
docker run -it --rm --gpus device=0 --ulimit memlock=-1 --ulimit stack=-1 --ulimit core=-1 --ipc=host --shm-size=32gb --name tritonproject -v $(pwd):/mnt tritonproject:cuda121gogo
```
### 구동 환경   
UBUNTU 20.04  
GPU: A100 DGX 80GB
torch==2.1.0a0+32f93b1  
cuda==12.2  
cudnn==8.9.0.5  

