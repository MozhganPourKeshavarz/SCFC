import json
import h5py
from shutil import copyfile
import numpy as np
import time

dst = "./my_output/"
src = "./flickr30k/"
LDA_dir = "./tuti/topics/"
prepared_files_dir = "./tuti/"
scores_dir = "./scores/"
analysis_score = []

OUTPUT = []
OUTPUT2 = []


def compute_cosine_similarity(base_vector, target_vector):
    """Compute the cosine similarity between two vectors based on the angular cosine distance
    return range -1 to 1, where 1 means two vectors are identical,
    -1 means reverse!*!, 0 means vectors are orthogonal
    where cosine(A,B) = dot(A,B) / ( || A || * || B || ) """

    np.seterr(all='print')
    cosine_similarity = 0

    try:
        base_vector = np.longdouble(base_vector)
        target_vector = np.longdouble(target_vector)
        vector_dot_products = np.dot(base_vector, target_vector)
        vector_norms = np.linalg.norm(base_vector) * np.linalg.norm(target_vector)
        cosine_similarity = np.divide(vector_dot_products, vector_norms)

        if vector_norms == 0.0:
            print
            'Error in vec in compute_cosine_similarity'
            print
            target_vector

    except:
        print("error")
        print(base_vector)
        print(target_vector)
        return 0

    return cosine_similarity + 1


def remove_outliers(candidates, column_indice=1, epsilon=0.15, number_of_nearest_neighbours=10):
    """ Remove outliers adaptively based on distance and a treshold value """

    remaining = len(candidates)
    visual_distance_scores = []

    try:
        for i in range(len(candidates)):
            visual_distance_scores.append(float(candidates[i][column_indice]))
    except Exception:
        print("Error in remove_outliers function")
        print()

    dist_min = min(visual_distance_scores)
    dist_max = max(visual_distance_scores)
    ind2remove = []
    # reverse the list, so that we can start removing from the furthest score
    candidates.sort(key=lambda c: c[column_indice], reverse=True)

    for i in range(len(candidates)):
        if float(candidates[i][column_indice]) > (1 + epsilon) * dist_min:
            if remaining > number_of_nearest_neighbours:  # Make sure we have at least some items left.
                ind2remove.append(i)
                remaining -= 1
            elif remaining == number_of_nearest_neighbours:
                break
    # candidates = np.delete(candidates, idx, axis=0) # remove outliers
    candidates = [x for i, x in enumerate(candidates) if i not in ind2remove]
    # candidates = candidates.tolist()
    candidates.reverse()
    # print(candidates)
    return candidates, dist_min, dist_max


#####load lda_topics#####

lda = h5py.File(LDA_dir + "lda_topics.h5", 'r+')

LDA_test_gt = list(lda["test_gt"])
LDA_test_pred = list(lda["test_pred"])

LDA_val_gt = list(lda["val_gt"])
LDA_val_pred = list(lda["val_pred"])

LDA_train_gt = list(lda["train_gt"])
LDA_train_pred = list(lda["train_pred"])

LDA_all_gt = LDA_train_gt + LDA_val_gt

#####load Image IDs####

# test img
with open(prepared_files_dir + "captions_test.json") as image_ids:
    test_image_ids = json.load(image_ids)  # 1000

test_img_list = list(test_image_ids["image_ids"])

# train img
with open(prepared_files_dir + "captions_train.json") as image_ids:
    train_image_ids = json.load(image_ids)  # 1000

train_img_list = list(train_image_ids["image_ids"])

# val img
with open(prepared_files_dir + "captions_val.json") as image_ids:
    val_image_ids = json.load(image_ids)  # 1000

val_img_list = list(val_image_ids["image_ids"])

all_img_list = train_img_list + val_img_list

# load regions CNN features
region_features = h5py.File(prepared_files_dir + "features_30res.h5", 'r+')  # test , train , val

f_train = region_features["train"]
f_val = region_features["val"]

f_train2 = np.array(f_train)
f_val2 = np.array(f_val)

all_region_features = np.concatenate((f_train2, f_val2), axis=0)

# load bbox
bbox = h5py.File(prepared_files_dir + "bbox.h5", 'r+')

bbox_train = bbox["train"]
bbox_test = bbox["test"]
bbox_val = bbox["val"]

b_train2 = np.array(bbox_train)
b_val2 = np.array(bbox_val)

all_bbox = np.concatenate((b_train2, b_val2), axis=0)

with open("entire_image.json") as entire:
    entire_image = json.load(entire)

test_entire_image = list(entire_image["test"])
train_entire_image = list(entire_image["train"])
val_entire_image = list(entire_image["val"])

all_entire_image = train_entire_image + val_entire_image


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


result = ""
for img in range(len(test_img_list)):
    print("image id:", img)
    t3 = time.time()
    # img=31
    retrived_img_list = []

    test_img_id = test_img_list[img]
    copyfile(src + test_img_id, dst + str(img) + "/" + "QUERY_IMG" + test_img_id)

    test_img_regions_score = []
    # print("Image Number : " , test_img_id)

    t1 = time.time()
    ####-----> remove
    # retrive regions scores
    score_file = open(scores_dir + str(test_img_id).split(".")[0] + ".txt", "r")
    lines = score_file.readlines()
    # print("step1")
    for row in lines:
        score = row.split("\n")[0].split("\t")[1]  # score
        test_img_regions_score.append(score)
    t2 = time.time()
    # print("time :",t2-t1)

    # retrive LDA for test img
    test_img_LDA = LDA_test_pred[img]

    # retrive regions CNN features
    test_img_regions_CNN = region_features["test"][img]  # (30 , 2048)

    # retrive bbox
    test_img_regions_bbox = bbox_test[img]

    score_matrix = np.zeros(shape=(30, 30))
    # score_matrix = [[0] * 30] * 30

    # find entire image (0,0,max_H,max_W)
    test_img_entire = test_entire_image[img]

    for candid_img_id in range(len(all_img_list)):
        # t5=time.time()
        # print("Candidate Image : ", all_img_list[candid_img_id])

        # retrive LDA for train img
        train_img_LDA = LDA_all_gt[candid_img_id]

        candid_img_regions_bbox = all_bbox[candid_img_id]

        candid_img_entire = all_entire_image[candid_img_id]

        candid_img = all_img_list[candid_img_id]
        candid_img_regions_CNN = all_region_features[candid_img_id]  # (30 , 2048)

        weighted_score = []
        t7 = time.time()
        for i in range(30):  # test_img_regions
            max_score_region = 0
            region_score = float(test_img_regions_score[i])
            test_img_region = test_img_regions_CNN[i]

            if list(test_img_regions_bbox[i]) != test_img_entire:
                for j in range(30):  # candid_img_regions

                    if list(candid_img_regions_bbox[j]) != candid_img_entire:
                        # print("Regions with saliency :)")
                        candid_img_region = candid_img_regions_CNN[j]
                        similarity = compute_cosine_similarity(test_img_region, candid_img_region)
                        score_matrix[i][j] = similarity
                        if similarity > max_score_region:
                            max_score_region = similarity

            else:

                similarity = compute_cosine_similarity(train_img_LDA, test_img_LDA)
                max_score_region = similarity
                region_score = 1

            weighted_similarity = max_score_region * region_score
            weighted_score.append(weighted_similarity)
        # t8=time.time()
        # print("t8-t7" , t8-t7)

        final_score = 0

        for scr in range(len(weighted_score)):
            final_score += weighted_score[scr]

        candid_eval = [candid_img, final_score]
        retrived_img_list.append(candid_eval)
        # t6=time.time()
        # print("t6-t5",t6-t5)

    # t9 = time.time()
    retrived_img_list.sort(key=lambda c: c[1], reverse=True)
    selected_retrived_img_list = retrived_img_list[:100]
    final_retrived, _, _ = remove_outliers(selected_retrived_img_list)
    # t10 = time.time()
    # print("t10-t9", t10 - t9)

    OUTPUT.append(final_retrived)

    # print("here")
    for item in range(len(final_retrived)):
        retrive_img_id = final_retrived[item][0]
        # print("copy image ...")
        copyfile(src + str(retrive_img_id),
                 dst + str(img) + "/" + str(final_retrived[item][1]) + "_" + str(retrive_img_id))

    # t4=time.time()
    # print("t4-t3" , t4-t3)
    # for candid_img_id in range(len(DATASET)):
    #
    #     if candid_img_id < len(train_img_list):
    #         candid_img = train_img_list[candid_img_id]
    #         candid_img_regions_CNN = region_features["train"][candid_img_id] #(30 , 2048)
    #         candid_img_regions_bbox = bbox_train[candid_img_id]
    #         weighted_score=[]
    #         for i in range(30): #test_img_regions
    #             max_score_region=0
    #             test_img_region = test_img_regions_CNN[i]
    #
    #             if list(test_img_regions_bbox[0][i])!=[0,0,max_H,max_w]:
    #                 for j in range(30): #candid_img_regions
    #                     candid_img_region = candid_img_regions_CNN[j]
    #                     similarity = compute_cosine_similarity(test_img_region , candid_img_region)
    #                     score_matrix[i][j]=similarity
    #                     if similarity>max_score_region:
    #                         max_score_region=similarity
    #
    #             else:
    #
    #
    #             #delete weight
    #             weighted_similarity = max_score_region
    #             weighted_score.append(weighted_similarity)
    #
    #         final_score=0
    #
    #         for scr in range(len(weighted_score)):
    #             final_score+=weighted_score[scr]
    #
    #         copyfile(src + candid_img , dst + str(img) + "/" + str(final_score.__format__(".5f")) +"_"+candid_img)
    #
    #     else:
    #         candid_img = val_img_list[candid_img_id]
    #         candid_img_regions_CNN = region_features["val"][candid_img_id]  # (30 , 2048)
    #         candid_img_regions_bbox = bbox_val[candid_img_id]
    #         weighted_score = []
    #         for i in range(30):  # test_img_regions
    #             max_score_region = 0
    #             test_img_region = test_img_regions_CNN[i]
    #
    #             for j in range(30):  # candid_img_regions
    #                 candid_img_region = candid_img_regions_CNN[j]
    #                 similarity = compute_cosine_similarity(test_img_region, candid_img_region)
    #                 score_matrix[i][j] = similarity
    #                 if similarity > max_score_region:
    #                     max_score_region = similarity
    #
    #
    #  delete weight
    #             weighted_similarity = max_score_region
    #             weighted_score.append(weighted_similarity)
    #
    #         final_score = 0
    #
    #         for scr in range(len(weighted_score)):
    #             final_score += weighted_score[scr]
    #
    #         copyfile(src + candid_img, dst + str(img) + "/" + str(final_score.__format__(".5f")) + "_" + candid_img)

    for q in OUTPUT[0]:
        z = []
        z.append(q[0])
        z.append(str(q[1]))
        OUTPUT2.append(z)

    # print("OUTPUT",OUTPUT[0])
    # print("len-OUTPUT",len(OUTPUT))

    RESULT = {"retrieved": OUTPUT2}

    with open('retrieved.json', 'w') as outfile:
        json.dump(RESULT, outfile)

