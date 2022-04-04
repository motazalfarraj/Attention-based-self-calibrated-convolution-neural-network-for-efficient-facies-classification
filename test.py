from core.loader.data_loader import *
from core.metrics import runningScore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("runs/Nov17_143300_SCPANet_skip/SCPANet_skip_model.pkl")
model = model.to(device)

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


pred1_i = np.zeros_like(labels1)
pred1_x = np.zeros_like(labels1)
pred2_i = np.zeros_like(labels2)
pred2_x = np.zeros_like(labels2)

seismic1, labels1 = transforms(seismic1, labels1)
seismic2, labels2 = transforms(seismic2, labels2)


#%%

with torch.no_grad():  # operations inside don't track history
    model.eval()
    for i in range(seismic1.shape[-2]):
        img, lbl = seismic1[...,i,:], labels1[...,i,:]
        img, lbl = img.to(device), lbl.to(device)
        outputs = model(img)
        pred = outputs.detach().max(1)[1].cpu().numpy()
        gt = lbl.detach().cpu().numpy()

        pred1_i[...,i,:] = np.squeeze(pred)
        running_metrics_test1.update(gt, pred)
        running_metrics_both.update(gt,pred)

    for i in range(seismic1.shape[-1]):
        img, lbl = seismic1[..., i], labels1[..., i]
        img, lbl = img.to(device), lbl.to(device)

        outputs = model(img)
        pred = outputs.detach().max(1)[1].cpu().numpy()
        gt = lbl.detach().cpu().numpy()

        pred1_x[..., i] = np.squeeze(pred)
        running_metrics_test1.update(gt, pred)
        running_metrics_both.update(gt,pred)

    for i in range(seismic2.shape[-2]):
        img, lbl = seismic2[...,i,:], labels2[...,i,:]
        img, lbl = img.to(device), lbl.to(device)
        outputs = model(img)
        pred = outputs.detach().max(1)[1].cpu().numpy()
        gt = lbl.detach().cpu().numpy()

        pred2_i[..., i,:] = np.squeeze(pred)
        running_metrics_test2.update(gt, pred)
        running_metrics_both.update(gt,pred)

    for i in range(seismic2.shape[-1]):
        img, lbl = seismic2[..., i], labels2[..., i]
        img, lbl = img.to(device), lbl.to(device)

        outputs = model(img)
        pred = outputs.detach().max(1)[1].cpu().numpy()
        gt = lbl.detach().cpu().numpy()

        pred2_x[..., i] = np.squeeze(pred)
        running_metrics_test2.update(gt, pred)
        running_metrics_both.update(gt,pred)

labels1 = np.squeeze(labels1.detach().cpu().numpy())
labels2 = np.squeeze(labels2.detach().cpu().numpy())

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
PA1 = (np.sum(labels1==pred1_i) + np.sum(labels1==pred1_x))/(2*np.prod(labels1.shape))
PA2 = (np.sum(labels2==pred2_i) + np.sum(labels2==pred2_x))/(2*np.prod(labels2.shape))
PA = (np.sum(labels2==pred2_i) + np.sum(labels2==pred2_x)+np.sum(labels1==pred1_i) + np.sum(labels1==pred1_x))/(2*np.prod(labels2.shape)+2*np.prod(labels1.shape))
print(PA1, PA2, PA)

#%%

#
# #%%
#         total_iteration = total_iteration + 1
#         image_original, labels_original = images, labels
#         images, labels = images.to(device), labels.to(device)
#
#         outputs = model(images)
#
#         pred = outputs.detach().max(1)[1].cpu().numpy()
#         gt = labels.detach().cpu().numpy()
#         running_metrics_split.update(gt, pred)
#         running_metrics_overall.update(gt, pred)
#
#         numbers = [0, 99, 149, 399, 499]
#
#         if i in numbers:
#             tb_original_image = vutils.make_grid(
#                 image_original[0][0], normalize=True, scale_each=True)
#             writer.add_image('test/original_image',
#                              tb_original_image, i)
#
#             labels_original = labels_original.numpy()[0]
#             correct_label_decoded = test_set.decode_segmap(np.squeeze(labels_original))
#             writer.add_image('test/original_label',
#                              np_to_tb(correct_label_decoded), i)
#             out = F.softmax(outputs, dim=1)
#
#             # this returns the max. channel number:
#             prediction = out.max(1)[1].cpu().numpy()[0]
#             # this returns the confidence:
#             confidence = out.max(1)[0].cpu().detach()[0]
#             tb_confidence = vutils.make_grid(
#                 confidence, normalize=True, scale_each=True)
#
#             decoded = test_set.decode_segmap(np.squeeze(prediction))
#             writer.add_image('test/predicted', np_to_tb(decoded), i)
#             writer.add_image('test/confidence', tb_confidence, i)
#
#             # uncomment if you want to visualize the different class heatmaps
#             unary = outputs.cpu().detach()
#             unary_max = torch.max(unary)
#             unary_min = torch.min(unary)
#             unary = unary.add((-1*unary_min))
#             unary = unary/(unary_max - unary_min)
#
#             for channel in range(0, len(class_names)):
#                 decoded_channel = unary[0][channel]
#                 tb_channel = vutils.make_grid(decoded_channel, normalize=True, scale_each=True)
#                 writer.add_image(f'test_classes/_{class_names[channel]}', tb_channel, i)
#
# # get scores and save in writer()
# score, class_iou = running_metrics_split.get_scores()
#
# # Add split results to TB:
# writer.add_text(f'test__{split}/',
#                 f'Pixel Acc: {score["Pixel Acc: "]:.3f}', 0)
# for cdx, class_name in enumerate(class_names):
#     writer.add_text(
#         f'test__{split}/', f'  {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}', 0)
#
# writer.add_text(
#     f'test__{split}/', f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}', 0)
# writer.add_text(
#     f'test__{split}/', f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}', 0)
# writer.add_text(f'test__{split}/',
#                 f'Mean IoU: {score["Mean IoU: "]:0.3f}', 0)
#
# running_metrics_split.reset()
#
# # FINAL TEST RESULTS:
# score, class_iou = running_metrics_overall.get_scores()
#
# # Add split results to TB:
# writer.add_text('test_final', f'Pixel Acc: {score["Pixel Acc: "]:.3f}', 0)
# for cdx, class_name in enumerate(class_names):
# writer.add_text(
#     'test_final', f'  {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}', 0)
#
# writer.add_text(
# 'test_final', f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}', 0)
# writer.add_text(
# 'test_final', f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}', 0)
# writer.add_text('test_final', f'Mean IoU: {score["Mean IoU: "]:0.3f}', 0)
#
# print('--------------- FINAL RESULTS -----------------')
# print(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
# for cdx, class_name in enumerate(class_names):
# print(
#     f'     {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}')
# print(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
# print(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
# print(f'Mean IoU: {score["Mean IoU: "]:0.3f}')
#
# # Save confusion matrix:
# confusion = score['confusion_matrix']
# np.savetxt(pjoin(log_dir,'confusion.csv'), confusion, delimiter=" ")
#
#
#
