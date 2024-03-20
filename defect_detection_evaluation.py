from __future__ import print_function
from FewShot_models.manipulate import *
from FewShot_models.training_parallel import *
from FewShot_models.imresize import imresize
import FewShot_models.functions as functions
import FewShot_models.models as models
from sklearn.metrics import roc_auc_score, accuracy_score, auc
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
import os, sys
import tarfile
from tqdm import tqdm
import urllib.request
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader


def defect_detection(input_name_model,test_size, opt):
    scale = int(opt.size_image)
    pos_class = opt.pos_class
    alpha = int(opt.alpha)
    path =  "mvtec_test_scale" + str(scale) + "_" + str(pos_class) + "_" + str(opt.num_images)
    if os.path.exists(path)==True:
        xTest_input = np.load(path + "/mvtec_data_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy")
        yTest_input = np.load(path + "/mvtec_labels_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy")
    else:
        if os.path.exists(path) == False:
            print("path not exists")
            exit()
    xTest_input = xTest_input[:test_size]
    yTest_input = yTest_input[:test_size]

    curr_heat_map_obj = None
    num_samples = xTest_input.shape[0]
    batch_size = 1
    batch_n = num_samples // batch_size
    opt.input_name = input_name_model
    opt.num_images = 1
    path = "TrainedModels/" + str(opt.input_name)[:-4] + \
           "/scale_factor=0.750000,alpha=" + str(alpha)
    probs_predictions = []
    real = torch.from_numpy(xTest_input[0]).cuda().unsqueeze(0)
    functions.adjust_scales2image(real, opt)
    scores_per_scale_dict = torch.from_numpy(np.zeros((opt.stop_scale+1,batch_n))).cuda()

    def compute_normalized_dict(scores_per_scale_dict):
        for scale in range(0, opt.stop_scale + 1):
            maxi = torch.max(scores_per_scale_dict[scale])
            mini = torch.min(scores_per_scale_dict[scale])
            scores_per_scale_dict[scale] = (scores_per_scale_dict[scale] - mini) / (maxi - mini)
        return scores_per_scale_dict

    transformations_list = np.load("TrainedModels/" + str(opt.input_name)[:-4] +  "/transformations.npy")

    with torch.no_grad():
        for i in range(batch_n):
            reals = {}
            real = torch.from_numpy(xTest_input[i]).unsqueeze(0).cuda()
            real = functions.norm(real)
            real = real[:, 0:3, :, :]
            functions.adjust_scales2image(real, opt)
            functions.adjust_scales2image(real, opt)
            # The resolution of a pictures is halved (opt.scale1=0.5)
            real = imresize(real, opt.scale1, opt)

            # Test
            # opt.gen_start_scale = 4
            # aa = functions.generate_in2coarsest(real, 5, 5,opt)

            # The resolution of a picture is further shrunk proportionally
            for index_image in range(int(opt.num_images)):
                reals[index_image] = []
                reals = functions.creat_reals_pyramid(real, reals, opt,index_image)

            err_total,err_total_avg, err_total_abs = [],[],[]
            for scale_num in range(0, opt.stop_scale+1, 1):
                opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), opt.size_image)
                opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), opt.size_image)
                netD = models.WDiscriminatorMulti(opt)
                if torch.cuda.device_count() > 1:
                    netD = DataParallelModel(netD, device_ids=opt.device_ids)
                netD.to(opt.device)
                netD.load_state_dict(torch.load('%s/%d/netD.pth' % (path, scale_num)))
                netD.eval()

                err_scale = []
                for index_image in range(int(opt.num_images)):

                    score_image_in_scale = 0
                    reals_transform = []
                    for index_transform, pair in enumerate(transformations_list):
                        real = reals[index_image][scale_num].to(opt.device)
                        curr_heat_map_obj = real
                        flag_color, is_flip, tx, ty, k_rotate = pair
                        real_augment = apply_augmentation(real, is_flip, tx, ty, k_rotate, flag_color).to(opt.device)
                        real_augment = torch.squeeze(real_augment)
                        reals_transform.append(real_augment)
                    real_transform = torch.stack(reals_transform)
                    output = netD(real_transform)
                    if isinstance(output, list):
                        output = [tens.to(opt.device) for tens in output]
                        output = torch.cat(output).detach()
                    else:
                        output = output.to(opt.device)
                    # (batch_size(# of Transformation): 54, channel: 55,
                    #  height, width)
                    original_hidden_input = output

                    # bbb = output.permute(0, 2, 3, 1).contiguous()
                    reshaped_output = output.permute(0, 2, 3, 1).contiguous()
                    shape = reshaped_output.shape
                    reshaped_output = reshaped_output.view(-1, shape[3])
                    # backup_a = reshaped_output
                    # Todo:
                    reshaped_output = reshaped_output[:, :opt.num_transforms]
                    # ===
                    # aaa = reshaped_output.reshape((shape[0], shape[1],
                    #                                shape[2], opt.num_transforms))
                    # =====
                    # Todo:
                    m = nn.Softmax(dim=1)
                    score_softmax = m(reshaped_output)
                    # ===
                    # score_softmax = reshaped_output
                    # =====
                    score_all = score_softmax.reshape(opt.num_transforms, -1,
                                                      opt.num_transforms)

                    heat_map = score_softmax.reshape((shape[0], shape[1],
                                                      shape[2], opt.num_transforms))
                    lst_heat_map = []
                    # for k in range(0, heat_map.shape[0]): # batch size
                    #     lst_heat_map.append([])
                    #     for l in range(0, heat_map.shape[1]): # height
                    #         lst_heat_map[k].append([])
                    #         for m in range(0, heat_map.shape[2]): # width
                    #             lst_heat_map[k][l].append([])
                    #             for n in range(0, heat_map.shape[3]): # channel
                    #                 lst_heat_map[k][l][m].append(
                    #                     heat_map[k][l][m][n].cpu().numpy())
                    # heat_map = torch.mean(heat_map, 3)

                    for k in range(0, heat_map.shape[0]):  # batch size
                        lst_heat_map.append([])
                        for l in range(0, heat_map.shape[1]):  # height
                            lst_heat_map[k].append([])
                            for m in range(0, heat_map.shape[2]):  # width
                                # lst_heat_map[k][l]\
                                #     .append(torch.mean(heat_map[k][l][m]).cpu().numpy())
                                    # heat_map[k][l][m][k].cpu().numpy())

                                lst_heat_map[k][l] \
                                    .append(heat_map[k][l][m][k].cpu().numpy())

                                # aaa = heat_map[k][l][m]
                                # print(heat_map[k][l][m][k])
                    heat_map = np.array(lst_heat_map)

                    # aaa = torch.tensor(heat_map)
                    # aaa = aaa.permute(0, 3, 1, 2)
                    # bbb = original_hidden_input[:,:-1,:,:]
                    # print(np.allclose(aaa[12], bbb[12].cpu().numpy(), rtol=0.01))


                    for j in range(opt.num_transforms):
                        current_transform = score_all[j]
                        score_transform = current_transform[:, j]
                        sorted_score_transform, indices = torch.sort(score_transform, descending=False, dim=0)
                        num_patches = int(sorted_score_transform.shape[0]* opt.fraction_defect)
                        score_transform = torch.mean(sorted_score_transform[:num_patches])
                        score_image_in_scale += score_transform
                    err_scale.append(score_image_in_scale)
                err_scale = torch.stack(err_scale)
                err = torch.max(err_scale, dim=0)[0]
                err = torch.mean(err).item()
                scores_per_scale_dict[scale_num][i] = (err)
                err_total.append(err)
                del netD
            avg_err_total = np.mean(err_total)
            probs_predictions.append(avg_err_total)


            agg_heat_map = np.mean(heat_map, 0)
            agg_heat_map = np.expand_dims(agg_heat_map, 0)
            agg_heat_map = np.expand_dims(agg_heat_map, 0)
            heat_map = torch.tensor(agg_heat_map)

            # Enlarge the heat map to size of the input image
            original_size = curr_heat_map_obj.shape[-1]
            recovered_heat_map = imresize(heat_map,
                                          original_size/heat_map.shape[-1],
                                          opt)
            maxi = torch.max(recovered_heat_map)
            mini = torch.min(recovered_heat_map)
            norm_heat_map = (recovered_heat_map - mini) / (maxi - mini)
            # aaa = norm_heat_map.cpu().numpy()

            functions.save_image(heat_map, curr_heat_map_obj.shape[-1],
                                 0, 0, "Visualization/block_heat_map_idx" +
                                 str(
                    i) +
                                 ".jpg")

            functions.save_image(norm_heat_map, curr_heat_map_obj.shape[-1],
                                 0, 0,
                                 "Visualization/recovered_heat_map_idx" + str(i) +
                                 ".jpg")

            functions.save_image(curr_heat_map_obj,
                                 curr_heat_map_obj.shape[-1],
                                 0, 0, "Visualization/origin_image_idx" + str(
                    i) +
                                 ".jpg")
            # functions.save_image(torch.tensor(xTest_input[i].cpu().numpy()).unsqueeze(0),
            #                      original_size,
            #                      0, 0, "Visualization/sample_idx" + str(i) +
            #                      ".jpg")


        print("Log is printed at:", opt.input_name + "_fraction_" + str(opt.fraction_defect) + ".txt")
        with open(opt.input_name + "_fraction_" + str(opt.fraction_defect) + ".txt", "w") as text_file:
            print(pos_class, "results: ", file=text_file)
            print(" ", file=text_file)
            print("Acc: ", file=text_file)
            print(yTest_input)
            print(probs_predictions)
            # acc = accuracy_score(yTest_input, probs_predictions)
            # print(acc)
            print("results without norm, without top_k: ", file=text_file)
            auc1 = roc_auc_score(yTest_input, probs_predictions)
            print("roc_auc_score (not normal) all ={}".format(auc1), file=text_file)
            scores_per_scale_dict_norm = compute_normalized_dict(scores_per_scale_dict)
            scores_per_scale_dict_norm = scores_per_scale_dict_norm.cpu().clone().numpy()
            print(" ", file=text_file)
            print("results with normalization ", file=text_file)
            probs_predictions_norm_all = np.mean(scores_per_scale_dict_norm, axis=0)
            auc1 = roc_auc_score(yTest_input, probs_predictions_norm_all)
            print("roc_auc_score T1 normalize all ={}".format(auc1), file=text_file)

    path = "mvtec_test_scale" + str(scale) + "_" + str(pos_class) + "_" + str(opt.num_images)
    os.remove(path + "/mvtec_data_test_" + str(pos_class) + str(scale) + "_" + str(opt.index_download) + ".npy")
    os.remove(path + "/mvtec_labels_test_" + str(pos_class) + str(scale) + "_" + str(opt.index_download) + ".npy")
    del xTest_input, yTest_input
