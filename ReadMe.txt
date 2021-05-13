# Colab에서 돌려볼 때
!pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
!pip3 install torchvision

from google.colab import drive
drive.mount('/content/gdrive/')
# 만약 그냥 마운트해서 할거라면 congfigs 파일 data_root경로 수정할 것. gdrive대신 그냥 drive로..(아래 둘 중 하나)
# '/content/gdrive/Shareddrives/zeogi_gogi/dataset/'
# '/content/drive/Shareddrives/zeogi_gogi/dataset/'

#제 깃으로 우선..
!git clone https://github.com/jjung-ah/ja.git

# 현재 위치 확인
!pwd
# 깃을 받은 경로로 이동
cd ja

# main 파일 실행
!python main.py