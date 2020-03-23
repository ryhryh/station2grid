python csv2npy.py --csv_path ./sat.csv --k 1 --weightKNN uniform
############### sat
declare -a k=("3" "5")
declare -a weightKNN=("uniform" "distance")

for i in "${k[@]}"
do
    for j in "${weightKNN[@]}"
    do
       python csv2npy.py --csv_path ./sat.csv --k "$i" --weightKNN "$j" 
    done
done



: '
############### air

#python csv2npy.py --csv_path ../csv/air.csv --k 1 --weightKNN uniform --thresholdKNN 500

declare -a k=("5" ) 
declare -a weightKNN=("uniform" "distance")

for i in "${k[@]}"
do
    for j in "${weightKNN[@]}"
    do
       python csv2npy.py --csv_path ../csv/air.csv --k "$i" --weightKNN "$j" --thresholdKNN 500
    done
done
'

