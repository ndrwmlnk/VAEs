import argparse
import os

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import _pickle as cPickle
import bz2
import cv2

from model_ae import ConvAE
from model_vae import ConvVAE
from model_sigma_vae import ConvSigmaVAE

## Arguments
#The only important argument is image_number which regulates how many images are rendered out
#all others are set automaitcally
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', type=str, default='optimal_sigma_vae', metavar='N',
                    help='which model to use: mse_vae,  gaussian_vae, or sigma_vae or optimal_sigma_vae')
parser.add_argument('--log_dir', type=str, default='test', metavar='N')
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--image_number', type=int, default=5000)
args = parser.parse_args()

## Cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

def video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 24, (width, height))

    print("Render Video")
    for idx, image in enumerate(tqdm(images)):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    images = np.zeros((args.image_number, 2304, 1920, 3), dtype="int8")#1728
    for model_idx, model in enumerate(["optimal_sigma_vae", "mse_vae", "ae"]):
        args.model = model
        for size_idx, size in enumerate([64, 16, 8, 16]):
            args.image_size = size
            #load cropped dataset
            if (size_idx == 3):
                test_data = bz2.BZ2File('dataset/test_crop_' + str(args.image_size) + 'x' + str(args.image_size) + '.pbz2', 'rb')
            # load scaled dataset
            else:
                test_data = bz2.BZ2File('dataset/test_' + str(args.image_size) + 'x' + str(args.image_size) + '.pbz2', 'rb')
            test_x = cPickle.load(test_data)
            test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_x))
            kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)
            #fill array with the original images
            for image_idx, data in enumerate(test_loader):
                if (image_idx < args.image_number):
                    data = np.stack(data).reshape(args.image_size, args.image_size, 3).astype(dtype="int8")
                    image_orig = Image.fromarray(data, "RGB").resize((192, 192), Image.NEAREST)
                    if(model_idx == 0 and size_idx == 0):
                        draw = ImageDraw.Draw(image_orig)
                        font = ImageFont.truetype("arial.ttf", 20)
                        draw.text((0, 165), "{:04d}".format(image_idx), (255, 255, 255), font=font)
                    image_orig = np.array(image_orig, dtype="int8")
                    x = (model_idx * 4 + size_idx) * 192
                    images[image_idx, x:(x+192), 0:192] = image_orig
            for dim_idx, dim in enumerate([256, 128, 64, 32, 16, 8, 4, 2, 1]):
                args.dim = dim
                args.log_dir = "{}_{}x{}_D{}".format(args.model, args.image_size, args.image_size, args.dim)
                print(args.log_dir)
                ## Build Model
                if (args.model == "ae"):
                    model = ConvAE(device, 3, args).to(device)
                if (args.model == "mse_vae"):
                    model = ConvVAE(device, 3, args).to(device)
                if (args.model == "optimal_sigma_vae"):
                    model = ConvSigmaVAE(device, 3, args).to(device)
                optimizer = optim.Adam(model.parameters(), lr=1e-4)
                model.load_state_dict(torch.load("checkpoints/{}/checkpoint_{}.pt".format(args.model, args.log_dir)))
                #fill array with reconstructions of model+size+dim
                for image_idx, data in enumerate(test_loader):
                    if(image_idx < args.image_number):
                        data = torch.Tensor((np.stack(data).reshape((args.batch_size, args.image_size, args.image_size, 3))) / 255).transpose(1, 2).transpose(1, 3)
                        image_recon, mu, logvar = model(data)
                        loss = torch.nn.MSELoss()(image_recon, data) * 100000
                        loss = loss.detach().cpu().numpy().astype(np.dtype(np.int))
                        image_recon = image_recon.detach().cpu().numpy()[0].transpose(1, 2, 0)
                        image_recon = (image_recon * 255).astype(dtype="int8")
                        img_temp = Image.fromarray(image_recon, "RGB").resize((192, 192), Image.NEAREST)
                        draw = ImageDraw.Draw(img_temp)
                        font = ImageFont.truetype("arial.ttf", 20)
                        draw.text((0, 165), "{:04d}".format(loss), (255, 255, 255), font=font)
                        img_temp = np.array(img_temp, dtype="int8")
                        x = (model_idx * 4 + size_idx) * 192
                        y = (dim_idx + 1) * 192
                        images[image_idx, x:(x+192), y:(y+192)] = img_temp
    #create images and save them
    os.makedirs('images', exist_ok=True)
    print("Save Images")
    for idx in tqdm(range(args.image_number)):
        image = Image.fromarray(images[idx], "RGB")
        image = image.save("images/{:05d}.png".format(idx))
    #video out of the image folder
    video("images/", "result.avi")
