import torch
import json
import torch
from torch import nn
import pickle as pickle
import h5py

# f = h5py.File('./prepaired_files/topics/lda_topics.h5', 'r')
# print(list(f.keys()))
#
# print(list(f["test_gt"][0]))
#
# exit()
#
#
#
#
#
#
#
# with open('./final_dataset/train36_imgid2idx.pkl', 'rb') as f:
#     x = pickle.load(f)
#
# print(x)
# exit()
#
#
# checkpoint_file = './pretrained_model/BEST_34checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
# checkpoint = torch.load(checkpoint_file)
# print(checkpoint)
# exit()
# a = torch.randn(32,14,14,2048, dtype=torch.double)
# image_features_mean = a.mean(1)
# print(image_features_mean.shape)

#
# karpathy_split = "./data/caption_datasets/dataset_coco.json"
#
# with open('./data/caption_datasets/dataset_coco.json') as data_file:
#     data = json.load(data_file)
#
#
# for userid in data["images"][0]:
#     print(userid)
#
# key = ["filepath",
# "sentids",
# "filename",
# "imgid",
# "split",
# "sentences",
# "cocoid"]
# print("-------------------------")
# for k in key:
#     print(data["images"][0][k])

full_att =nn.Linear(2, 1)

m1 = torch.ones(2, 1)
m2 = torch.ones(3, 5)

print(m1)
print("------")
m3= m1.unsqueeze(0)
print(m3)
m4=m1.unsqueeze(1)
print("------")
print(m4)

print("*******************")

# att1= torch.Tensor(4, 3, 2)
att1 = torch.randint(2, 10, (4, 3, 2))
print("att1" , att1)

# att2= torch.Tensor(4,2).randint_(0, 10)
att2 = torch.randint(2, 10, (4, 2))
att3 = torch.randint(2,10, (4,2))
print("att2",att2)
print("att3",att3)
mul = att2*att3
print("mul", mul)
exit()

# print("att2" , att2)

att2_=att2.unsqueeze(1)

print("att2_unsquueze(1)" , att2_)

# a= att1+att2_
a= att1*att2_


print("a ", a)
exit()

print(att1 )
print(att1.shape )
print("---")

print(att2_ )
print(att2_.shape )
print("---")

print(a )
print(a.shape )
print("---")

ful = full_att(att1 + att2_).squeeze(2)
print(ful)
print(ful.shape)