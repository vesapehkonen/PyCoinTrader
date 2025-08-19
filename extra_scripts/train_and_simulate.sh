set -e
set -x

rm -rf ~/tmp2/tests/*

run_simulate() {
    ./clean_output.sh
    python train_models.py ../data/input/train_data.json.tmp
    python run_simulate.py ../data/input/simu_data.json.tmp
}

# Merge any number of JSON array files (in ../data/input) into OUTFILE
merge_input() {
  local outfile="$1"; shift         # first arg = output filename
  ( cd ../data/input && jq -s 'add' "$@" > "$outfile" )
}

merge_input train_data.json.tmp part1.json part2.json part3.json part4.json part5.json part6.json part7.json 
merge_input simu_data.json.tmp  part8.json part9.json
run_simulate
