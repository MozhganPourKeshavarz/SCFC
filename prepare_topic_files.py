import h5py
import json


LDA_dir="./topic/"
prepared_files = "./prepared_files/"



#####load lda_topics#####

lda = h5py.File(LDA_dir + "lda_topics.h5", 'r+')

LDA_test_gt = list(lda["test_gt"])
print(len(LDA_test_gt))

LDA_val_gt = list(lda["val_gt"])
print(len(LDA_val_gt))

LDA_train_gt = list(lda["train_gt"])
print(len(LDA_train_gt))




#####load Image IDs####

# test img
with open(prepared_files+ "captions_test.json") as image_ids:
    test_image_ids = json.load(image_ids)  # 1000

test_img_list = list(test_image_ids["image_ids"])
# print(len(test_img_list))

# train img
with open(prepared_files + "captions_train.json") as image_ids:
    train_image_ids = json.load(image_ids)  # 1000

train_img_list = list(train_image_ids["image_ids"])
# print(len(train_img_list))

# val img
with open(prepared_files + "captions_val.json") as image_ids:
    val_image_ids = json.load(image_ids)  # 1000

val_img_list = list(val_image_ids["image_ids"])
# print(len(val_img_list))


TEST_lda_dict= {}
TRAIN_lda_dict= {}
VAL_lda_dict= {}

for img in range(len(test_img_list)):
    TEST_lda_dict.update( {img : LDA_test_gt[img].tolist() } )

for img in range(len(train_img_list)):
    TRAIN_lda_dict.update({ img:LDA_train_gt[img].tolist() })

for img in range(len(val_img_list)):
    VAL_lda_dict.update({img:LDA_val_gt[img].tolist()})


coco_dict = {
"train" : TRAIN_lda_dict ,
"test" : TEST_lda_dict ,
"val" : VAL_lda_dict
}



with open('./topic/topics.json', 'w') as outfile:
    json.dump(coco_dict, outfile)
