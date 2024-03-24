# 스터디 참여를 위한 RoPE 구현  
본 코드를 바탕으로 openai triton을 활용하여 Fused Rotary Embedding을 구현해보고 CUDA 커널을 활용한 Fused Rotary Embedding 구현 함수와 PyTorch 기반으로 짜여진 Fused Rotary Embedding과의 속도 비교를 수행합니다.  
순서는 코드 소개, 코드 실행 방법, 벤치마크 결과 분석, 향후 개선 방향 순서로 구성되어 있습니다.  

## 코드 소개  
### triton_rotary_main.py  
RopeFowardAndBackward 클래스에서 triton rotary embedding의 forward 메소드와 backward 메소드를 정의한다.  
forward 메소드와 backward 메소드는 매우 유사한데 set_rotary_kernel 메소드에서 backward=True이냐 False이냐에 따라 달라진다.  
backward=True면 커널 내부에서 forward에서 쓰이는 sin 부호가 -sin으로 음수로 바뀐다.  
### triton_rotary_kernel.py  
set_rotary_kernel 함수에서 그리드 크기 설정과 같은 kernel 실행을 위한 환경을 설정하고, rotary_kernel 함수에서 실질적으로 triton 상의 커널 연산 수행을 한다.  
### pytest_benchmark_cuda_triton_comparison.py  
test_fused_rope 함수에서 triton fused kernel을 활용한 rope, cuda fused kernel을 활용한 rope, pytorch의 rope를 연산한 후 서로 output을 torch.testing.assert_close를 활용하여 상호 비교한다.
또한 각자 방법별 rope output의 gradient를 연산한 후 위와 같이 torch.testing.assert_close를 활용하여 상호 비교한다.  
### benchmark  
pytest_benchmark_cuda_triton_comparison.py 내부에 벤치마크 함수들을 구현해 두었다.  (위와 같은 함수)  
벤치마크를 통해 Triton Fused RoPE 수행 속도와 Cuda Fused RoPE 수행 속도를 비교한다.  
그리고 Triton Fused RoPE 수행 속도와 Torch RoPE(unfused)의 수행 속도를 비교한다.  
seq_length, hidden_size, head_num, batch_size를 기준으로 벤치마크를 수행한다.  
A100 80GB에서 실행되어서 변수들의 크기가 증가할 시 OOM이 발생할 수 있음을 유의한다.   

## 코드 실행 방법  
### 구동 환경   
UBUNTU 20.04  
GPU: A100 DGX 80GB  
torch==2.1.0a0+32f93b1  
cuda==12.2  
cudnn==8.9.0.5  
triton==2.1.0  
Nvidia TransformerEngine(https://github.com/NVIDIA/TransformerEngine) 활용  

#### 1. 도커 파일을 활용한 빌드 
```
docker build -f tritonProject.Dockerfile -t tritonproject:cuda121gogo .
```
#### 2. 도커 컨테이너 실행하기
```
docker run -it --rm --gpus device=0 --ulimit memlock=-1 --ulimit stack=-1 --ulimit core=-1 --ipc=host --shm-size=32gb --name tritonproject -v $(pwd):/mnt tritonproject:cuda121gogo
```
#### 3. 유닛 테스트 수행  

```
pytest pytest_benchmark_cuda_triton_comparison.py -s
```
#### 4. 벤치마크 수행   
```
python pytest_benchmark_cuda_triton_comparison.py
```

## 벤치마크 결과 분석  



