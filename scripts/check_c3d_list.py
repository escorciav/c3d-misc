from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

import pandas as pd


def main(txt_file, output_flag, check_all_frames, t_res, imgfmt, layer,
         no_stop):
    """Report if all the inputs OR outputs exist"""
    idx_table = {'video': 0, 'f-init': 1}
    df = pd.read_csv(txt_file, sep=' ', header=None)

    check_fn = os.path.isdir
    if output_flag:
        check_fn = os.path.isfile

    for i, row in df.iterrows():
        video = row[idx_table['video']]
        if output_flag:
            video = video + layer

        if not check_fn(video):
            print(video)
            if not no_stop:
                break

        if check_all_frames and not output_flag:
            f_init = row[idx_table['f-init']]
            for j in range(f_init, f_init + t_res):
                imgfile = os.path.join(video, imgfmt.format(j))
                if not os.path.isfile(imgfile):
                    print(imgfile)
                    break


if __name__ == '__main__':
    description = 'Check missing inputs (frames) or outputs (features)'
    p = ArgumentParser(description=description,
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--txt-file', required=True,
                   help='Input/Output list given to extract features')
    p.add_argument('-o', '--output-flag', action='store_true',
                   help='txt-file is an output list')
    p.add_argument('-c', '--check-all-frames', action='store_false',
                   help='Ensure all frames requested by a line are there')
    p.add_argument('-l', '--t-res', default=16, type=int,
                   help='temporal length of the clips')
    p.add_argument('-f', '--imgfmt', default='{0:06d}.png',
                   help='Image format')
    p.add_argument('-of', '--layer', default='.fc6-1',
                   help='Extracted layer')
    p.add_argument('-ns', '--no-stop', action='store_true',
                   help='Nop stop at first error')

    main(**vars(p.parse_args()))
