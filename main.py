import numpy as np
import PIL
import cv2
from PIL import Image
from dlr import DLRModel
import io
import matplotlib.image as mpimg
import json
import random
from matplotlib import pyplot as plt
import time
import os

with open('cardbot_synset.txt') as f:
    synset = eval(f.read())
threshold = .25
dlr_model = DLRModel(model_path='model', dev_type='gpu')


def transform_image(img):
    #start with CV2 image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    # Now convert to PIL
    pimg = Image.fromarray(img)

    pimg = np.array(pimg) - np.array([123., 117., 104.])
    pimg /= np.array([58.395, 57.12, 57.375])
    pimg = pimg.transpose((2,0,1))
    pimg = pimg[np.newaxis, :]
    return pimg


def infer(t):
    response = {'prediction':[]}
    # Execute

    input_data = {'data': t}
    output = dlr_model.run(input_data)

    #objects = output.get_output(0)
    #scores = output.get_output(1)
    #bboxes = output.get_output(2)
    objects = output[0]
    scores = output[1]
    bboxes = output[2]

    objects_list = objects.tolist()
    score_list = scores.tolist()
    bboxes_list = bboxes.tolist()

    for x in objects_list[0]:
        response['prediction'].append(x)
    for idx, score in enumerate(score_list[0]):
        response['prediction'][idx].append(score[0])
    print(response)
    
    for idx, bbox in enumerate(bboxes_list[0]):
        for x in bbox:
            response['prediction'][idx].append(x)

    #for index, x in enumerate(np_scores[0]):
    #    if x[0] > .80:

            #prediction = '[' + str(synset[int(np_objects[0][index][0])]) + ", "
            #prediction += str(np_bboxes[0][index])
            #prediction += "]"

            #response['prediction'].append(prediction)
            #print(synset[int(np_objects[0][index][0])], x[0], np_bboxes[0][index])
    response_body = json.dumps(response)
    return response_body


def visualize(index, img_path, response):
    img = Image.open(img_path)
    img = img.resize((512, 512), Image.BILINEAR)
    img_bytes = io.BytesIO()
    img.save(img_bytes, 'JPEG')
    right_sized = mpimg.imread(img_bytes, 'jpg')
    f = io.BytesIO
    plt.clf()
    plt.imshow(right_sized)
    colors = dict()
    for pred in response:
        (id, score, x0, y0, x1, y1) = pred
        if id > -1 and id < 52:
            rs = synset[id]
            if score >= threshold:
                cls_id = int(id)
                if cls_id not in colors:
                    colors[cls_id] = (random.random(), random.random(), random.random())
                xmin = int(x0)
                ymin = int(y0)
                xmax = int(x1)
                ymax = int(y1)
                rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax-ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=3.5)
                plt.gca().add_patch(rect)
                plt.gca().text(xmin, ymin - 2,
                               '{:s} {:.3f}'.format(rs, score),
                               bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                               fontsize=12, color='white')
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    fig1 = plt.gcf()
    fig1.set_size_inches(4, 4)
    fig1.savefig("results/results" + str(index) + ".png", format='png', bbox_inches='tight', transparent=True,
                 pad_inches=0, dpi=100)


def orchestrate():
    # Take all images in Observation and run inference
    # Write Images out to results folder with bounding boxes and predictions
    # Prepare report that lists inference times and mean inference
    index = 0
    for r, d, f in os.walk("observations"):
        for file in f:
            img_path = 'observations/'+ file
            start = time.time()
            img = cv2.imread(img_path)
            t = transform_image(img)
            response = infer(t)
            stop = time.time()
            duration = stop-start
            #TODO - create inference time report
            print("{}:{}".format(index, duration))
            j = json.loads(response)
            visualize(index, img_path, j['prediction'])
            index += 1



def main():
    orchestrate()


if __name__ == '__main__':
    main()