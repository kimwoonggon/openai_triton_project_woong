Openai triton project for visual transformer implementation  
Dockerfile Build  
```
docker build -f tritonProject.Dockerfile -t tritonproject:cuda121 .
```
Docker container start
```
docker run -it --rm --gpus device=0 --ulimit memlock=-1 --ulimit stack=-1 --ulimit core=-1 --ipc=host --shm-size=32gb -p 7891:7891 --name tritonproject -v $(pwd):/mnt tritonproject:cuda121
```
