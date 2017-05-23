# C3D Miscellaneous

A couple of scripts to interact smoothly with [C3D project](https://github.com/facebook/C3D).

## Citation

If you find any piece of code valuable for your research please cite this work:

```
@Inbook{Escorcia2016,
author="Escorcia, Victor and Caba Heilbron, Fabian and Niebles, Juan Carlos and Ghanem, Bernard",
editor="Leibe, Bastian and Matas, Jiri and Sebe, Nicu and Welling, Max",
title="DAPs: Deep Action Proposals for Action Understanding",
bookTitle="Computer Vision -- ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III",
year="2016",
publisher="Springer International Publishing",
address="Cham",
pages="768--784",
isbn="978-3-319-46487-9",
doi="10.1007/978-3-319-46487-9_47",
url="http://dx.doi.org/10.1007/978-3-319-46487-9_47"
}
```

If you are not in the academia, you can also support us by giving us a :star: in the github banner.

## What can you find?

- [Generate list for C3D feature extraction](clip_generation.py).
  This program will help you to extract C3D densely over a video.
  You can control how densely you want the features. Even, the temporal receptive filed of the C3D,

  BTW, it partially generates the list that you need. I usually do the rest with [this bash script](scripts/format_list.sh), sorry I like bash.

- [Pack binaries with C3D features into HDF5](dump_hdf5.py).
  If you extract features for many videos, you will get a lot of binaries.
  If you place all the C3D of a video into a unique folder, we will pack them into a HDF5 file for you.
  It can handle multiple features for the same video, take a look a the help of that program.

## How to install it?

We haven't packed it yet. Clone the repo and use it on your demand :wink:.

The programs here are only utilities to generate inputs for C3D binaries or digest its outputs.
There is *not* library link between C3D library and programs here.

## No Bro, How to Install C3D?

We create a handy conda environment to install C3D with only three requirements, as opposed to > 6 requirements of that project.
Take a look of our recipe in [this repo](https://github.com/escorciav/C3D/tree/build-conda)
