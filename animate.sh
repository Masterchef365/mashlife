dir=animation
frames=60
rm $dir/*.ppm
cargo build --release
mkdir $dir 2>/dev/null
for i in {0..5000..100}; do
    ./target/release/mashlife --steps $i --outfile "$dir/out_$i.ppm" $1
done

#pushd $dir

