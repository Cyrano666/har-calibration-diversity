from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'fitdeps'),str(HERE/'torchdeps')]
for key in ['NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='4'
os.environ['NUMBA_CACHE_DIR']=str(HERE/'numba_cache');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class Classic:
 def __init__(self,name,seed):self.name=name;self.seed=seed
 def fit(self,X,Z,y):
  if self.name=='MR':
   from aeon.transformations.collection.convolution_based import MiniRocket
   self.mean=X.mean(axis=(0,2),keepdims=True);self.std=np.maximum(X.std(axis=(0,2),keepdims=True),1e-6)
   self.transform=MiniRocket(n_kernels=2016,n_jobs=4,random_state=self.seed)
   z=self.transform.fit_transform(((X-self.mean)/self.std).astype('float32'))
  else:z=Z
  self.scaler=StandardScaler();z=self.scaler.fit_transform(z);self.model=LogisticRegression(C=1,max_iter=2000,tol=1e-4,solver='lbfgs',random_state=self.seed)
  self.model.fit(z,y);self.classes_=self.model.classes_;return self
 def proba(self,X,Z):
  z=self.transform.transform(((X-self.mean)/self.std).astype('float32')) if self.name=='MR' else Z
  return self.model.predict_proba(self.scaler.transform(z))

class InceptionModel:
 def __init__(self,seed):self.seed=seed
 def fit(self,X,Z,y):
  import torch
  from torch import nn
  torch.set_num_threads(4);torch.manual_seed(self.seed);torch.cuda.manual_seed_all(self.seed)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.use_deterministic_algorithms(True)
  assert torch.cuda.is_available(),'CUDA is required for the declared IT training run'
  class Module(nn.Module):
   def __init__(self,channels):
    super().__init__();self.bottleneck=nn.Conv1d(channels,32,1,bias=False);self.branches=nn.ModuleList([nn.Conv1d(32,32,k,padding=k//2,bias=False) for k in [39,19,9]]);self.pool=nn.Sequential(nn.MaxPool1d(3,stride=1,padding=1),nn.Conv1d(channels,32,1,bias=False));self.norm=nn.BatchNorm1d(128)
   def forward(self,x):
    z=self.bottleneck(x);return torch.relu(self.norm(torch.cat([b(z) for b in self.branches]+[self.pool(x)],dim=1)))
  class Net(nn.Module):
   def __init__(self,K):
    super().__init__();self.blocks=nn.ModuleList([Module(3 if i==0 else 128) for i in range(6)]);self.shortcuts=nn.ModuleList([nn.Sequential(nn.Conv1d(3 if i==0 else 128,128,1,bias=False),nn.BatchNorm1d(128)) for i in range(2)]);self.head=nn.Linear(128,K)
   def forward(self,x):
    residual=x
    for i,block in enumerate(self.blocks):
     x=block(x)
     if i%3==2:x=torch.relu(x+self.shortcuts[i//3](residual));residual=x
    return self.head(x.mean(dim=2))
  self.classes_=np.unique(y);mapped=np.searchsorted(self.classes_,y);self.mean=X.mean(axis=(0,2),keepdims=True);self.std=np.maximum(X.std(axis=(0,2),keepdims=True),1e-6)
  x=torch.as_tensor(((X-self.mean)/self.std).astype('float32'),device='cuda');target=torch.as_tensor(mapped,dtype=torch.long,device='cuda')
  self.net=Net(len(self.classes_)).cuda();optimizer=torch.optim.Adam(self.net.parameters(),lr=.001);loss_fn=nn.CrossEntropyLoss()
  self.epoch_losses=[]
  for epoch in range(40):
   self.net.train();perm=torch.randperm(len(x),device='cuda');total=0.
   for left in range(0,len(x),128):
    ix=perm[left:left+128];optimizer.zero_grad(set_to_none=True);loss=loss_fn(self.net(x[ix]),target[ix]);loss.backward();optimizer.step();total+=loss.detach().item()*len(ix)
   self.epoch_losses.append(total/len(x))
  self.net.eval();self.parameters=sum(p.numel() for p in self.net.parameters());return self
 def proba(self,X,Z):
  import torch
  rows=[]
  with torch.no_grad():
   for left in range(0,len(X),512):
    x=torch.as_tensor(((X[left:left+512]-self.mean)/self.std).astype('float32'),device='cuda');rows.append(self.net(x).softmax(dim=1).cpu().numpy())
  return np.concatenate(rows)

def create_model(name,seed):return InceptionModel(seed) if name=='IT' else Classic(name,seed)
