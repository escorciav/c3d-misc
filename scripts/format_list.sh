# ----------------------------------------------------------------------------
# Format input and oputput list used by c3d binaries
#
# Usage: format_list.sh [input-almost-ready] [name-output-list] \
#                       [prefix-in] [prefix-out]
#
# 1. Create output list from input list.
# 2. Preppend path to (a) input list and (b) output list.
#
# Tip: Do you have many gpus? split the lists with
# $ split -n l/[num-gpus] -d [list] [prefix]
# ----------------------------------------------------------------------------
# 1. Generate output list
awk '{printf "%s/%06d\n", $1, $2}' < $1 > $2

# 2. Add prefix at the begining of each line
prefix_in=$3
sed -i -e "s&^&${prefix_in}&" $1

prefix_out=$4
sed -i -e "s&^&${prefix_out}&" $2
