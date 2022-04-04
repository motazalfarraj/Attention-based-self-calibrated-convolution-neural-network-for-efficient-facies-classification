import torch
from core.models.ViTNet import mydnn_18
from thop import clever_format
from thop import profile

model = mydnn_18(num_classes=6).cuda(2)
input = torch.randn(1, 1,256,256).cuda(2)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")

print(macs, params)

out = model(input)
print(out.shape)

#%%




# import numpy as np
# import torch
#
# from core.loader.data_loader import *
# from core.metrics import runningScore
# from thop import clever_format
# from thop import profile
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# from core.models.SRNet import Res34Unetv4
# from core.models.section_deconvnet import section_deconvnet_skip
# from core.models.SCPANet import SCPANet_skip
#
# #%%
# model1_path = "runs/Feb01_122245_section_deconvnet_skip/section_deconvnet_skip_model.pkl"
# model2_path = "runs/Nov17_143300_SCPANet_skip/SCPANet_skip_model.pkl"
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# trained_model1 = torch.load(model1_path).module.state_dict()
# trained_model2 = torch.load(model2_path).module.state_dict()
# #%%
# model1 = section_deconvnet_skip(n_classes=6)
# model2 = SCPANet_skip(n_classes=6)
#
# model1.load_state_dict(trained_model1, strict=True)
# model2.load_state_dict(trained_model2, strict=True)
#
# #%%
# model1.to(device)
# model2.to(device)
#
# input = torch.randn(1, 1,256,256).to(device)
# macs1, params1 = profile(model1.module, inputs=(input, ))
# macs1, params1 = clever_format([macs1, params1], "%.3f")
#
# macs2, params2 = profile(model2.module, inputs=(input, ))
# macs2, params2 = clever_format([macs2, params2], "%.3f")
#
# print(macs1, params1)
# print(macs2, params2)
#
# #%%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model1 = torch.load(model1_path).eval()
# model2 = torch.load(model2_path).eval()
#
#
#
# def compute_metrics(ref, pred, n_classes=6):
#     ref_flat = np.squeeze(ref).reshape(-1,1)
#     pred_flat = np.squeeze(pred).reshape(-1,1)
#     c_matrix = confusion_matrix(ref_flat, pred_flat, labels=np.arange(n_classes))
#     PA = np.sum(np.diag(c_matrix))/np.sum(c_matrix)
#     CA = np.diag(c_matrix) / c_matrix.sum(axis=1)
#     IU = np.diag(c_matrix)/(c_matrix.sum(axis=1)+c_matrix.sum(axis=0)-np.diag(c_matrix))
#     return PA, CA, IU, c_matrix
#
#
# seismic1 = np.moveaxis(np.load("data/dutch/test_once/test1_seismic.npy"), -1, 0)
# seismic2 = np.moveaxis(np.load("data/dutch/test_once/test2_seismic.npy"), -1, 0)
#
# labels1 = np.moveaxis(np.load("data/dutch/test_once/test1_labels.npy"), -1, 0)
# labels2 = np.moveaxis(np.load("data/dutch/test_once/test2_labels.npy"), -1, 0)
#
# def transforms(seismic, labels):
#     mean = 0.000941  # average of the training data
#     seismic = seismic-mean
#     seismic = np.expand_dims(np.expand_dims(seismic, 0), 0)
#     labels = np.expand_dims(np.expand_dims(labels, 0), 0)
#     seismic = torch.from_numpy(seismic)
#     seismic = seismic.float()
#     labels = torch.from_numpy(labels)
#     labels = labels.long()
#
#     return seismic, labels
#
#
# pred1_i = np.zeros_like(labels1)
# pred1_x = np.zeros_like(labels1)
# pred2_i = np.zeros_like(labels2)
# pred2_x = np.zeros_like(labels2)
#
# seismic1, labels1 = transforms(seismic1, labels1)
# seismic2, labels2 = transforms(seismic2, labels2)
#
# #%%
# def per_section_results(model, seismic, labels):
#     PA_i = []
#     PA_x = []
#     CA_i = []
#     CA_x = []
#     IU_i = []
#     IU_x = []
#     with torch.no_grad():  # operations inside don't track history
#         model.eval()
#         for i in range(seismic.shape[-2]):
#             img, lbl = seismic[..., i, :], labels1[..., i, :]
#             img, lbl = img.to(device), lbl.to(device)
#             outputs = model(img)
#             pred = outputs.detach().max(1)[1].cpu().numpy()
#             gt = lbl.detach().cpu().numpy()
#             pred1_i[..., i, :] = np.squeeze(pred)
#             PA, CA, IU, c_matrix = compute_metrics(gt, pred)
#             PA_i.append(PA)
#             CA_i.append(CA)
#             IU_i.append(IU)
#
#
#         for i in range(seismic.shape[-1]):
#             img, lbl = seismic[..., i], labels1[..., i]
#             img, lbl = img.to(device), lbl.to(device)
#             outputs = model(img)
#             pred = outputs.detach().max(1)[1].cpu().numpy()
#             gt = lbl.detach().cpu().numpy()
#             pred1_x[..., i] = np.squeeze(pred)
#             PA, CA, IU, c_matrix = compute_metrics(gt, pred)
#             PA_x.append(PA)
#             CA_x.append(CA)
#             IU_x.append(IU)
#
#     return PA_i, PA_x, CA_i,CA_x, IU_i, IU_x
#
#
#
# #%%
# PA1_i, PA1_x, CA1_i,CA1_x, IU1_i, IU1_x = per_section_results(model1,seismic1, labels1)
# PA2_i, PA2_x, CA2_i,CA2_x, IU2_i, IU2_x = per_section_results(model2,seismic1, labels1)
#
# plt.plot(PA1_i)
# plt.plot(PA2_i)
# plt.legend(["Model1", "Model2"])
# plt.show()
#
# plt.plot(PA1_x)
# plt.plot(PA2_x)
# plt.legend(["Model1", "Model2"])
# plt.show()
#
# #%%
# PA1_i, PA1_x, CA1_i,CA1_x, IU1_i, IU1_x = per_section_results(model1,seismic2, labels2)
# PA2_i, PA2_x, CA2_i,CA2_x, IU2_i, IU2_x = per_section_results(model2,seismic2, labels2)
# plt.plot(PA1_i)
# plt.plot(PA2_i)
# plt.legend(["Model1", "Model2"])
# plt.show()
#
# plt.plot(PA1_x)
# plt.plot(PA2_x)
# plt.legend(["Model1", "Model2"])
# plt.show()
#
#
# #%%
# def get_results(model, seismic1, labels1, seismic2, labels2):
#     running_metrics_test1 = runningScore(6)
#     running_metrics_test2 = runningScore(6)
#     running_metrics_all= runningScore(6)
#     with torch.no_grad():  # operations inside don't track history
#         model.eval()
#         for i in range(seismic1.shape[-2]):
#             img, lbl = seismic1[..., i, :], labels1[..., i, :]
#             img, lbl = img.to(device), lbl.to(device)
#             outputs = model(img)
#             pred = outputs.detach().max(1)[1].cpu().numpy()
#             gt = lbl.detach().cpu().numpy()
#
#             pred1_i[..., i, :] = np.squeeze(pred)
#             running_metrics_test1.update(gt, pred)
#             running_metrics_all.update(gt, pred)
#
#         for i in range(seismic1.shape[-1]):
#             img, lbl = seismic1[..., i], labels1[..., i]
#             img, lbl = img.to(device), lbl.to(device)
#
#             outputs = model(img)
#             pred = outputs.detach().max(1)[1].cpu().numpy()
#             gt = lbl.detach().cpu().numpy()
#
#             pred1_x[..., i] = np.squeeze(pred)
#             running_metrics_test1.update(gt, pred)
#             running_metrics_all.update(gt, pred)
#
#         for i in range(seismic2.shape[-2]):
#             img, lbl = seismic2[..., i, :], labels2[..., i, :]
#             img, lbl = img.to(device), lbl.to(device)
#             outputs = model(img)
#             pred = outputs.detach().max(1)[1].cpu().numpy()
#             gt = lbl.detach().cpu().numpy()
#
#             pred2_i[..., i, :] = np.squeeze(pred)
#             running_metrics_test2.update(gt, pred)
#             running_metrics_all.update(gt, pred)
#
#         for i in range(seismic2.shape[-1]):
#             img, lbl = seismic2[..., i], labels2[..., i]
#             img, lbl = img.to(device), lbl.to(device)
#
#             outputs = model(img)
#             pred = outputs.detach().max(1)[1].cpu().numpy()
#             gt = lbl.detach().cpu().numpy()
#
#             pred2_x[..., i] = np.squeeze(pred)
#             running_metrics_test2.update(gt, pred)
#             running_metrics_all.update(gt, pred)
#
#     return running_metrics_test1, running_metrics_test2, running_metrics_all
#
# def display_results(running_metrics):
#     for metrics in running_metrics:
#         score, class_iou = metrics.get_scores()
#         print('--------------- Test -----------------')
#         print(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
#         print(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
#         print(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
#         print(f'Mean IoU: {score["Mean IoU: "]:0.3f}')
#
# running_metrics = get_results(model1, seismic1, labels1, seismic2, labels2)
# display_results(running_metrics)
#
# #%%
#
#
#
#
