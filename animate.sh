dir=animation
frames=60
cargo build --release
mkdir $dir 2>/dev/null
for i in {2..60}; do
    ./target/release/mashlife --steps $i --outfile "$dir/out_$i.ppm" $1
done

#pushd $dir

