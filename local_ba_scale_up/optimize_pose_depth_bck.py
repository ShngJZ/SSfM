import os, sys, subprocess, glob, natsort
import argparse
import random
import time

proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, proj_root)

def acquire_split_path(args):
    split_export_path = os.path.join(
        proj_root, 'split', 'scannet', 'scannet.txt'
    )
    return split_export_path

def rnd_sel_seq_to_gen(args):
    split_export_path = acquire_split_path(args)
    with open(split_export_path) as file:
        entries = file.readlines()
    random.seed(os.getpid())
    random.shuffle(entries)

    seq_for_generation, entry_for_generation = None, None
    for entry in entries:
        seq, rgbroot, rgb1, rgb2, rgb3, rgb4 = entry.rstrip('\n').split(' ')
        output_root = os.path.join(proj_root, 'checkpoint', args.train_module, 'subset_{}'.format(args.train_sub), seq, args.train_name)
        if os.path.exists(output_root):
            continue
        else:
            os.makedirs(output_root, exist_ok=True)
            seq_for_generation, entry_for_generation = seq, entry
            break

    if (seq_for_generation is None) and (entry_for_generation is None):
        # Sleep One Minute
        time.sleep(60)
        for entry in entries:
            seq, rgbroot, rgb1, rgb2, rgb3, rgb4 = entry.rstrip('\n').split(' ')
            output_root = os.path.join(
                proj_root, 'checkpoint', args.train_module, 'subset_{}'.format(args.train_sub), seq, args.train_name
            )
            contents = glob.glob(os.path.join(output_root, "*"))
            if len(contents) == 0:
                seq_for_generation, entry_for_generation = seq, entry
                print("{} Has Folder with Empty Content After Waiting One Minute".format(seq_for_generation))
                print("Regenerate.....")
                break

    return seq_for_generation, entry_for_generation

def is_all_generated(args):
    split_export_path = acquire_split_path(args)
    with open(split_export_path) as file:
        entries = file.readlines()

    all_generated = True
    for entry in entries:
        seq, rgbroot, rgb1, rgb2, rgb3, rgb4 = entry.rstrip('\n').split(' ')
        output_root = os.path.join(proj_root, 'checkpoint', args.train_module, 'subset_{}'.format(args.train_sub), seq, args.train_name)

        nerf_ckpt_paths = glob.glob(os.path.join(output_root, '*.pth.tar'))
        nerf_ckpt_paths = natsort.natsorted(nerf_ckpt_paths)
        if len(nerf_ckpt_paths) == 0:
            all_generated = False
            print("Entry %s not generated" % entry.rstrip("\n"))
            continue
        nerf_ckpt_path = nerf_ckpt_paths[-1]
        nerf_ckpt_path_iter = int(nerf_ckpt_path.split('/')[-1].split('.')[0].split('-')[1])

        if int(nerf_ckpt_path_iter) < 80000:
            all_generated = False
            print("Entry %s not generated" % entry.rstrip("\n"))
            continue

    return all_generated

def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--train_module', type=str, help='Name of module in the "train_settings/" folder.', default="joint_pose_nerf_training/scannet_depth_exp")
    parser.add_argument('--train_name', type=str, help='Name of the train settings file.', default="zoedepth_pdcnet")
    parser.add_argument('--data_root', type=str, default='/home/ubuntu/disk6/RSfM-Datasets/ScanNet',
                        help='Name of the train settings file.')
    parser.add_argument('--dataset', type=str, choices=["scannet", "kitti360"], default="scannet")

    # arguments
    parser.add_argument('--scene', type=str, default=None,
                        help='scene')
    parser.add_argument('--dataset_entry', type=str, default=None,
                        help='dataset_entry')
    parser.add_argument('--train_sub', type=int, default=5,
                        help='train subset: how many input views to consider?')
    args = parser.parse_args()

    while True:
        seq, dataset_entry = rnd_sel_seq_to_gen(args)
        if seq is None:
            break
        else:
            print("Generate on Scene %s. \nEntry: %s" % (seq, dataset_entry))
            args.scene = seq
            args.dataset_entry = dataset_entry

            print("=============Stage 1 Pose Optimization. =============")
            executable = sys.executable
            return_stat = subprocess.call(
                "{} {}/run_trainval.py {} {} --scene {} --dataset_entry {} --train_sub {} --data_root {} --stage 1".format(
                    executable, proj_root, args.train_module, args.train_name, args.scene, "'{}'".format(args.dataset_entry), args.train_sub, args.data_root
                ), shell=True)
            assert return_stat == 0

            print("=============Stage 2 Depth Optimization. =============")
            return_stat = subprocess.call(
                "{} {}/run_trainval.py {} {} --scene {} --dataset_entry {} --train_sub {} --data_root {} --stage 2".format(
                    executable, proj_root, args.train_module, args.train_name, args.scene, "'{}'".format(args.dataset_entry), args.train_sub, args.data_root
                ), shell=True)
            assert return_stat == 0

    if is_all_generated(args):
        executable = sys.executable

        return_stat = subprocess.call(
            "{} {}/evaluation/eval_nerf_depth.py {} {} --train_sub {} --th_number 2 --relative_count 0 --dense 1".format(
                executable, proj_root, args.train_module, args.train_name, args.train_sub
            ), shell=True)
        assert return_stat == 0

        return_stat = subprocess.call(
            "{} {}/evaluation/eval_nerf_depth.py {} {} --train_sub {} --th_number 2 --relative_count 0".format(
                executable, proj_root, args.train_module, args.train_name, args.train_sub
            ), shell=True)
        assert return_stat == 0

        return_stat = subprocess.call(
            "{} {}/evaluation/eval_nerf_depth.py {} {} --train_sub {} --th_number 3 --relative_count 0".format(
                executable, proj_root, args.train_module, args.train_name, args.train_sub
            ), shell=True)
        assert return_stat == 0

        return_stat = subprocess.call(
            "{} {}/evaluation/eval_nerf_depth.py {} {} --train_sub {} --th_number 2 --relative_count 1".format(
                executable, proj_root, args.train_module, args.train_name, args.train_sub
            ), shell=True)
        assert return_stat == 0

        return_stat = subprocess.call(
            "{} {}/evaluation/eval_nerf_depth.py {} {} --train_sub {} --th_number 2 --relative_count 2".format(
                executable, proj_root, args.train_module, args.train_name, args.train_sub
            ), shell=True)
        assert return_stat == 0

        return_stat = subprocess.call(
            "{} {}/evaluation/eval_pose_by_corres.py {} {} --train_sub {} --pose_eval pose_optimized".format(
                executable, proj_root, args.train_module, args.train_name, args.train_sub
            ), shell=True)
        assert return_stat == 0

        return_stat = subprocess.call(
            "{} {}/evaluation/eval_pose_by_3dcons.py {} {} --train_sub {} --pose_eval pose_optimized".format(
                executable, proj_root, args.train_module, args.train_name, args.train_sub
            ), shell=True)
        assert return_stat == 0

        return_stat = subprocess.call(
            "{} {}/evaluation/eval_pose_by_pose.py {} {} --train_sub {} --pose_eval pose_optimized".format(
                executable, proj_root, args.train_module, args.train_name, args.train_sub
            ), shell=True)
        assert return_stat == 0

if __name__ == '__main__':
    main()