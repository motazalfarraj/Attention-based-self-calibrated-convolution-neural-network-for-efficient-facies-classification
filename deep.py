from core.models.ms_models.resnet_unet import Res34Unetv4
import torch
from core.metrics import runningScore
import numpy as np
from torch.nn.functional import pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Res34Unetv4(n_classes=6)
trained_model = torch.load("core/deep_seismic/Weights/dutchf3_seresnetunet_patch_section_depth.pth")
trained_model = {k.replace("module.", ""): v for (k, v) in trained_model.items()}
model.load_state_dict(trained_model, strict=True)
model.cuda()

#%%
running_metrics_test1 = runningScore(6)
running_metrics_test2 = runningScore(6)
running_metrics_both = runningScore(6)

seismic1 = np.moveaxis(np.load("data/dutch/test_once/test1_seismic.npy"), -1, 0)
seismic2 = np.moveaxis(np.load("data/dutch/test_once/test2_seismic.npy"), -1, 0)

labels1 = np.moveaxis(np.load("data/dutch/test_once/test1_labels.npy"), -1, 0)
labels2 = np.moveaxis(np.load("data/dutch/test_once/test2_labels.npy"), -1, 0)

def transforms(seismic, labels):
    mean = 0.000941  # average of the training data
    seismic = seismic-mean
    seismic = np.expand_dims(np.expand_dims(seismic, 0), 0)
    labels = np.expand_dims(np.expand_dims(labels, 0), 0)
    seismic = torch.from_numpy(seismic)
    seismic = seismic.float()
    labels = torch.from_numpy(labels)
    labels = labels.long()

    return seismic, labels



seismic1, labels1 = transforms(seismic1, labels1)
seismic2, labels2 = transforms(seismic2, labels2)


#%%

s_vol1 = pad(seismic1.squeeze(0), (0,67,0,56,0,1), "constant", 0)
l_vol1 = pad(labels1.squeeze(0), (0,67,0,56,0,1), "constant", 255)
s_vol1 = torch.cat(torch.split(s_vol1,256,dim=-1), dim=0)
l_vol1 = torch.cat(torch.split(l_vol1,256,dim=-1), dim=0)

s_vol2 = pad(seismic2.squeeze(0), (0,56,0,167,0,1), "constant", 0)
l_vol2 = pad(labels2.squeeze(0), (0,56,0,167,0,1), "constant", 255)
s_vol2 = torch.cat(torch.split(s_vol2,256,dim=-2), dim=0)
l_vol2 = torch.cat(torch.split(l_vol2,256,dim=-2), dim=0)

s_vol = torch.cat((s_vol1,s_vol2))
l_vol = torch.cat((l_vol1,l_vol2))

s_vol = torch.cat((s_vol.unsqueeze(1),s_vol.unsqueeze(1),s_vol.unsqueeze(1)), dim=1)

#%%
pred_i = np.zeros_like(l_vol)
pred_x = np.zeros_like(l_vol)

with torch.no_grad():
    model.eval()
    for i in range(l_vol.shape[-2]):
        img, lbl = s_vol[...,i,:], l_vol[...,i,:]
        img, lbl = img.to(device), lbl.to(device)
        outputs = model(img)
        pred = outputs.detach().max(1)[1].cpu().numpy()
        gt = lbl.detach().cpu().numpy()

        pred_i[...,i,:] = np.squeeze(pred)
        running_metrics_test1.update(gt, pred)
        running_metrics_both.update(gt,pred)

    for i in range(l_vol.shape[-1]):
        img, lbl = s_vol[..., i], l_vol[..., i]
        img, lbl = img.to(device), lbl.to(device)

        outputs = model(img)
        pred = outputs.detach().max(1)[1].cpu().numpy()
        gt = lbl.detach().cpu().numpy()

        pred_x[..., i] = np.squeeze(pred)
        running_metrics_test1.update(gt, pred)
        running_metrics_both.update(gt,pred)

#%%
score, class_iou = running_metrics_test1.get_scores()
print('--------------- Test 1 -----------------')
print(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
print(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
print(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
print(f'Mean IoU: {score["Mean IoU: "]:0.3f}')

score, class_iou = running_metrics_test2.get_scores()
print('--------------- Test 2 -----------------')
print(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
print(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
print(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
print(f'Mean IoU: {score["Mean IoU: "]:0.3f}')

score, class_iou = running_metrics_both.get_scores()
print('--------------- Both -----------------')
print(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
print(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
print(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
print(f'Mean IoU: {score["Mean IoU: "]:0.3f}')


#%%
from sklearn.metrics import confusion_matrix
ref_flat = np.squeeze(l_vol).reshape(-1, 1)
pred_flat = np.squeeze(pred_x).reshape(-1, 1)
c_matrix = confusion_matrix(ref_flat, pred_flat, labels=np.arange(6))
PA = np.sum(np.diag(c_matrix)) / np.sum(c_matrix)
CA = np.diag(c_matrix) / c_matrix.sum(axis=1)
IU = np.diag(c_matrix) / (c_matrix.sum(axis=1) + c_matrix.sum(axis=0) - np.diag(c_matrix))
