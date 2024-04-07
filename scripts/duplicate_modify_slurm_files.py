# duplicate the slurm_snowlines_SITE-ID.bash file and save with new site ID

old_id = "SITE-ID"
new_ids = ["01.00570", "01.01104", "01.09162", "01.10851", "01.14523",
           "01.22207", "01.22699", "02.05157", "02.06859", "02.12721",
           "02.12722", "02.13130", "02.14297", "02.16722", "02.17023",
           "02.18778", "01.01180", "01.01415", "01.23658", "02.00259",
           "02.02340"]

# Load input file
input_fn = "slurm_RGI60-SITE-ID.bash"
with open(input_fn, "r") as input_file:
    content = input_file.read()

# Iterate over new_ids
for new_id in new_ids:

    # Perform the replacement
    new_content = content.replace(old_id, new_id)

    # Define output file name
    output_fn = input_fn.replace(old_id, new_id)

    # Open the input file for writing
    with open(output_fn, "w") as output_file:
        output_file.write(new_content)
    print("Saved new bash file for " + new_id)
