import mne
import numpy as np
from psg_ann import Psg_Ann
from mne.io import read_raw_edf
import os
import glob
# from dataset import rewrite
class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}
npz_path=r"E:\eason\project\deep learning\WTV\load_dataset/data_edf_20_npz"

def edf_read(data_path,rewrite=False,npz_path=npz_path):
    psg_fnames = glob.glob(os.path.join(data_path, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(data_path, "*Hypnogram.edf"))
    # 获取路径
    # path = os.path.abspath(psg_fnames[0])
    # basepath = os.path.dirname(path)
    # rewrite=True
    # rewrite = False
    if not rewrite:
        # 读取npz文件
        try:
            dataset_x,dataset_y=get_npz(npz_path)
            return dataset_x,dataset_y
        except:
            print("npz do not exist")
    # 如果重写
    if rewrite:
        print("读取edf文件,共{}".format(len(psg_fnames)))
        for i in range(len(psg_fnames)):
        # for i in range(5):
            #读取edf文件
            raw=read_raw_edf(psg_fnames[i], preload=False, verbose=False)
            sampling_rate = raw.info['sfreq']
            signal_dict={}

            if sampling_rate not in signal_dict: signal_dict[sampling_rate] = []
            signal_dict[sampling_rate].append((raw.ch_names, raw.get_data()))
            # Wrap data into DigitalSignals
            for sfreq, signal_lists in signal_dict.items():
                data = np.concatenate([x for _, x in signal_lists], axis=0)
                # data = np.transpose(data)
                channel_names = [name for names, _ in signal_lists for name in names]
            data = data[0]  # 取第一个通道数据 即EEG Fpz-Cz
            # Read the Annotations
            # consist of sleep stages W, R, 1, 2, 3, 4, M (Movement time) and ? (not scored).
            mne_anno:mne.Annotations=mne.read_annotations(ann_fnames[i])
            labels=list(sorted(set(mne_anno.description)))
            interval,annotation=[],[]
            label2int={lb:i for i,lb in enumerate(labels)}
            print(mne_anno)
            for onset,duration,label in zip(mne_anno.onset,mne_anno.duration,mne_anno.description):
                interval.append((onset,onset+duration))
                # annotation.append(label2int[label])
                annotation.append(ann2label[label])
            #
            # digital_signals.append(Psg_Ann(data,annotation,interval))
            process_signal=Psg_Ann(data,annotation,interval)

            # Saving as numpy files
            filename = os.path.basename(psg_fnames[i]).replace(".edf", "(raw).npz")
            save_dict = {
                "x": process_signal.data_process,
                "y": process_signal.label,
                "fs": sampling_rate
            }
            # np.savez(os.path.join(basepath,filename), **save_dict)
            np.savez(os.path.join(npz_path,filename), **save_dict)
            print(" ---------- have saved the {} file in {} ---------".format(filename,npz_path))
        dataset_x,dataset_y=get_npz(npz_path)
        print("  ----------have loaded all files----------")
        return dataset_x,dataset_y


def get_npz(npz_path):
    file_name_list = []
    file_list = os.listdir(npz_path)
    for file_name in file_list:
        if file_name == "dataset.npz":
            dataset = np.load(os.path.join(npz_path,file_name))
            dataset_x = dataset["x"]
            dataset_y = dataset["y"]
            return dataset_x, dataset_y

        if file_name[-4:] == ".npz" and file_name != "dataset.npz":
            file_name_list.append(os.path.join(npz_path, file_name))
            # 融合数组
    dataset = np.load(file_name_list[0])
    dataset_x = dataset["x"]
    dataset_y = dataset["y"]
    for file_name in file_name_list[1::]:
        data = np.load(file_name)
        data_x = data["x"]
        data_y = data["y"]
        dataset_x = np.r_[dataset_x, data_x]
        dataset_y = np.r_[dataset_y, data_y]
    np.savez(os.path.join(npz_path, 'dataset.npz'), x=dataset_x, y=dataset_y)
    return dataset_x, dataset_y

if __name__ == "__main__":
    path=r"E:\workstation\reference\xai-sleep\data\sleepedf"
    dir_path1=r"E:\workstation\reference\xai-sleep\data\sleepedf\SC4001E0-PSG.npz"
    dir_path2 = r"E:\workstation\reference\xai-sleep\data\sleepedf\SC4002E0-PSG.npz"
    dir_path3 = r"E:\workstation\reference\xai-sleep\data\sleepedf\SC4011E0-PSG.npz"
    dataset1 = np.load(dir_path1)
    dataset2 = np.load(dir_path2)
    dataset3 = np.load(dir_path3)
    dataset_y1 = dataset1["y"]
    dataset_y2 = dataset2["y"]
    print(dataset_y2)
    # x,y=get_npz(path)
    # print(x)