import matplotlib.pyplot as plt
import h5py
import numpy as np
import math
import sys, time
import os



"read .hd5 file and convert to ndarray"
def read_h5(path):
    f = h5py.File(path,"r")
    data = f["/data"]
    return np.array(data)

def normal(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def renormal(data,target):
    target_max = np.max(target)
    target_min = np.min(target)
    return (data) * (target_max - target_min) + target_min

# Interpolation kernel   插值内核
def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

#Paddnig
def padding(img,H,W,C):
    zimg = np.zeros((H+4,W+4,C))
    zimg[2:H+2,2:W+2,:C] = img
    #Pad the first/last two col and row
    zimg[2:H+2,0:2,:C]=img[:,0:1,:C]
    zimg[H+2:H+4,2:W+2,:]=img[H-1:H,:,:]
    zimg[2:H+2,W+2:W+4,:]=img[:,W-1:W,:]
    zimg[0:2,2:W+2,:C]=img[0:1,:,:C]
    #Pad the missing eight points
    zimg[0:2,0:2,:C]=img[0,0,:C]
    zimg[H+2:H+4,0:2,:C]=img[H-1,0,:C]
    zimg[H+2:H+4,W+2:W+4,:C]=img[H-1,W-1,:C]
    zimg[0:2,W+2:W+4,:C]=img[0,W-1,:C]
    return zimg

def get_progressbar_str(progress):
    END = 170
    MAX_LEN = 30
    BAR_LEN = int(MAX_LEN * progress)
    return ('Progress:[' + '=' * BAR_LEN +
            ('>' if BAR_LEN < MAX_LEN else '') +
            ' ' * (MAX_LEN - BAR_LEN) +
            '] %.1f%%' % (progress * 100.))


# Bicubic operation   双三次插值
def bicubic(img, ratio, a):
    # Get image size
    H, W, C = img.shape

    img = padding(img, H, W, C)
    # Create new image
    dH = math.floor(H * ratio)
    dW = math.floor(W * ratio)
    dst = np.zeros((dH, dW, C))

    h = 1 / ratio

    print('Start bicubic interpolation')
    print('It will take a little while...')
    inc = 0
    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                x, y = i * h + 2, j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                mat_l = np.matrix([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])
                mat_m = np.matrix([[img[int(y - y1), int(x - x1), c], img[int(y - y2), int(x - x1), c],
                                    img[int(y + y3), int(x - x1), c], img[int(y + y4), int(x - x1), c]],
                                   [img[int(y - y1), int(x - x2), c], img[int(y - y2), int(x - x2), c],
                                    img[int(y + y3), int(x - x2), c], img[int(y + y4), int(x - x2), c]],
                                   [img[int(y - y1), int(x + x3), c], img[int(y - y2), int(x + x3), c],
                                    img[int(y + y3), int(x + x3), c], img[int(y + y4), int(x + x3), c]],
                                   [img[int(y - y1), int(x + x4), c], img[int(y - y2), int(x + x4), c],
                                    img[int(y + y3), int(x + x4), c], img[int(y + y4), int(x + x4), c]]])
                mat_r = np.matrix([[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

                # Print progress
                inc = inc + 1
                sys.stderr.write('\r\033[K' + get_progressbar_str(inc / (C * dH * dW)))
                sys.stderr.flush()
    sys.stderr.write('\n')
    sys.stderr.flush()
    return dst

def bicubic_1c(img, ratio, a):
    img = np.expand_dims(img, axis=2)
    dst = bicubic(img, ratio, a)
    dst = np.squeeze(dst, axis=2)
    return dst


def fig():
    low1 = read_h5("./k3_crossline_401_240×400.h5").T
    low2 = read_h5("./k3_inline_17_676×350.h5").T
    high1 = read_h5("./predcited/k3_crossline_401_240×400.h5").T
    high1 = renormal(high1,low1)
    high2 = read_h5("./predcited/k3_inline_17_676×350.h5").T
    high2 = renormal(high2,low2)
    # plt.figure()
    fig,ax = plt.subplots(2,2)
    im = ax[0][0].imshow(low1,cmap="seismic")
    # plt.colorbar()
    # plt.subplot(222)
    im = ax[0][1].imshow(high1, cmap="seismic")
    # plt.colorbar()
    # plt.subplot(223)
    im = ax[1][0].imshow(low2, cmap="seismic")
    # plt.colorbar()
    # plt.subplot(224)
    im = ax[1][1].imshow(high2, cmap="seismic")
    fig.colorbar(im,ax=[ax[0][0]])
    fig.colorbar(im, ax=[ax[0][1]])
    fig.colorbar(im, ax=[ax[1][0]])
    fig.colorbar(im, ax=[ax[1][1]])
    plt.show()


def plt_compare(data,target):
    plt.subplot(221)
    plt.imshow(data,cmap="seismic")
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(target, cmap="seismic")
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(data[50:100,50:100], cmap="seismic")
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(target[100:200,100:200], cmap="seismic")
    plt.colorbar()
    plt.show()


def show_result(data,target,top_left_x,top_left_y,width,height,edgecolor = "red",linewidth = 1):

    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'normal',
            'size': 12,
            }

    font1 = {'family': 'Times New Roman',
             'color': 'black',
             'weight': 'normal',
             'size': 14,
             }
    t_min = np.min(target)
    t_max = np.max(target)

    fig,ax = plt.subplots(2,3,figsize = (12,8))
    plt.subplots_adjust(left = 0.1,right = 0.95,bottom = 0.1,top = 0.9,wspace = 0.2,hspace=0.2)
    # define rectangle
    rect = plt.Rectangle((top_left_x, top_left_y), width, height, fill=False, edgecolor=edgecolor,
                         linewidth=linewidth)
    rect_up_bic = plt.Rectangle((top_left_x * 2, top_left_y * 2), width * 2, height * 2, fill=False,
                                edgecolor=edgecolor, linewidth=linewidth )
    rect_up_gan = plt.Rectangle((top_left_x * 2, top_left_y * 2), width * 2, height * 2, fill=False,
                                edgecolor=edgecolor,
                                linewidth=linewidth )
    text_x = -8
    text_y = -16
    shape_x,shape_y = data.shape
    #### complete figure
    # original
    ax1 = ax[0][0].imshow(data,cmap="seismic",aspect="auto",vmin = t_min,vmax = t_max)
    ax[0][0].add_patch(rect)
    ax[0][0].text(text_y, text_x, "(a)", font1)
    # ax[0][0].tick_params(labelsize=8)
    ax[0][0].set_title("Original sesimic image",font)
    ax[0][0].set_ylabel("Samples", font)


    # bicubic
    ratio = 2
    a = -0.5
    bic_data = bicubic_1c(data,ratio,a)
    ax2 = ax[0][1].imshow(bic_data, cmap="seismic",aspect="auto",vmin = t_min,vmax = t_max)
    ax[0][1].add_patch(rect_up_bic)
    ax[0][1].text(text_y * 2,text_x * 2, "(b)", font1)
    ax[0][1].set_title("Reconstruct by bicubic interpolation", font)


    # GAN
    ax3 = ax[0][2].imshow(target, cmap="seismic",aspect="auto",vmin = t_min,vmax = t_max)
    # print(target.shape)
    ax[0][2].add_patch(rect_up_gan)
    ax[0][2].text(text_y * 2,text_x * 2, "(c)", font1)
    ax[0][2].set_title("Reconstruct by our GAN", font)
    # fig.colorbar(ax3, ax=[ax[0][0],ax[0][1],ax[0][2]])

    #### subpatch
    # original
    original_sub = data[top_left_y:top_left_y + height,top_left_x:top_left_x + width]
    ax4 = ax[1][0].imshow(original_sub, cmap="seismic",aspect="auto",vmin = t_min,vmax = t_max)
    ax[1][0].spines['bottom'].set_color(edgecolor)
    ax[1][0].spines['top'].set_color(edgecolor)
    ax[1][0].spines['left'].set_color(edgecolor)
    ax[1][0].spines['right'].set_color(edgecolor)
    ax[1][0].text(text_y * width  / shape_y,text_x * height / shape_x, "(d)", font1)
    ax[1][0].set_ylabel("Samples", font)
    ax[1][0].set_xlabel("Traces", font)



    # bicubic
    bic_data_sub = bic_data[top_left_y * 2:top_left_y * 2+ height * 2, top_left_x * 2:top_left_x * 2+ width * 2]
    ax5 = ax[1][1].imshow(bic_data_sub, cmap="seismic",aspect="auto",vmin = t_min,vmax = t_max)
    ax[1][1].spines['bottom'].set_color(edgecolor)
    ax[1][1].spines['top'].set_color(edgecolor)
    ax[1][1].spines['left'].set_color(edgecolor)
    ax[1][1].spines['right'].set_color(edgecolor)
    ax[1][1].text(text_y * width  / shape_y * 2,text_x * height / shape_x * 2, "(e)", font1)
    ax[1][1].set_xlabel("Traces", font)

    # GAN
    target_sub = target[top_left_y * 2:top_left_y * 2+ height * 2, top_left_x * 2:top_left_x * 2+ width * 2]
    ax6 = ax[1][2].imshow(target_sub, cmap="seismic",aspect="auto",vmin = t_min,vmax = t_max)
    ax[1][2].spines['bottom'].set_color(edgecolor)
    ax[1][2].spines['top'].set_color(edgecolor)
    ax[1][2].spines['left'].set_color(edgecolor)
    ax[1][2].spines['right'].set_color(edgecolor)
    ax[1][2].text(text_y * width  / shape_y * 2,text_x * height / shape_x * 2, "(f)", font1)
    ax[1][2].set_xlabel("Traces", font)
    # fig.colorbar(ax3, ax=[ax[0][0],ax[0][1],ax[0][2]],fraction=0.046)
    fig.colorbar(ax6, ax=[ax[1][0], ax[1][1], ax[1][2],ax[0][0],ax[0][1],ax[0][2]], fraction=0.046)

    plt.savefig("./resultwu_lr.png",dpi = 100)




def plt_result():
    data = read_h5("./tp_352×240.h5").T
    target = read_h5("./predict/predicted/tp_352×240.h5").T
    # target = normal(target)
    # target = renormal(target,data)
    show_result(data,target,top_left_x = 20,top_left_y = 100,height = 100,width = 200,edgecolor = "green")





if __name__ == "__main__":
    plt_result()
