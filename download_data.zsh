#!/usr/bin/zsh
# The script downloads wind speed data in 80 m height starting from startmonth.startyear up to endmonth.endyear,
# crops it to the box specified in (lat1, lon1, lat2, lon2; cf. cdo) and aggregates the files

lat1=47.40724 # source: https://latitudelongitude.org/de/
lat2=54.9079
lon1=5.98815
lon2=14.98853

startyear=2009
endyear=2019
startmonth=9
endmonth=8

for y in {$startyear..$endyear};
do 
	for m in {1..12};
	do		
		if [[ (($y -gt $startyear) || ($m -ge $startmonth)) && (($y -lt $endyear) || ($m -le $endmonth)) ]]; then
			mzero=$(printf "%02d" $m)
			filename=WS_080m.2D.$y$mzero.nc4
			# Download file from DWD data archive
			wget https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/converted/hourly/2D/WS_080/$filename
			# Crop to bounding box of Germany using cdo
			cdo sellonlatbox,$lon1,$lon2,$lat1,$lat2 $filename SEL_$filename
			# Remove complete file to reduce disk space
			rm $filename
		fi
		
	done
done

# Aggregate Data
cdo mergetime SEL_*.nc4 WS_080m.2D.$startyear$startmonth-$endyear$endmonth.aggregated.nc4

# Remove SEL files
rm SEL_*.nc4