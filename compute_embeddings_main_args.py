import argparse
from coil_core.compute_embeddings import execute
import numpy as np

def resave_rep_datasets(rep_datasets_path):
    perception_rep_dataset, speed_rep_dataset, intentions_rep_dataset = np.load(rep_datasets_path)
    perception_rep_dataset = np.asarray(perception_rep_dataset.tolist(), dtype=np.float32)
    speed_rep_dataset = np.asarray(speed_rep_dataset.tolist(), dtype=np.float32)
    intentions_rep_dataset = np.asarray(intentions_rep_dataset.tolist(), dtype=np.float32)
    np.save(rep_datasets_path.rsplit('_', 1)[0] + '_perception_rep.npy', perception_rep_dataset)
    np.save(rep_datasets_path.rsplit('_', 1)[0] + '_speed_rep.npy', speed_rep_dataset)
    np.save(rep_datasets_path.rsplit('_', 1)[0] + '_intentions_rep.npy', intentions_rep_dataset)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--gpu',
        dest='gpu',
        type=str
    )
    argparser.add_argument(
        '-f',
        '--folder',
        type=str
    )
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )
    argparser.add_argument(
        '-vd',
        '--val-dataset',
        dest='val_dataset',
        type=str,
        default=''
    )
    argparser.add_argument(
        '-d',
        '--dataset',
        type=str,
        default=''
    )
    args = argparser.parse_args()
    # Compute embeddings for validation dataset
    if args.val_dataset != '':
        execute(args.gpu, args.folder, args.exp, args.val_dataset, True)
        print("Done with val set")
        resave_rep_datasets('_preloads/' + args.folder + '_' + args.exp + '_' + args.val_dataset + '_representations.npy')
        print("Done with val set")
    # Compute embeddings for train dataset
    if args.dataset != '':
        execute(args.gpu, args.folder, args.exp, args.dataset)
        print("Done with train set")
        resave_rep_datasets('_preloads/' + args.folder + '_' + args.exp + '_' + args.dataset + '_representations.npy')
        print("Done with train set")
