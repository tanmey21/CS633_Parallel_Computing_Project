#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);

int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

// Define global domain and process grid
int nx = atoi(argv[4]), ny = atoi(argv[5]), nz = atoi(argv[6]);
int px = atoi(argv[1]), py = atoi(argv[2]), pz = atoi(argv[3]);
int nc = atoi(argv[7]);

if (px * py * pz != size) {
    if (rank == 0) printf("Error: px * py * pz must equal number of MPI processes\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
}

// Create Cartesian topology
int dims[3] = {px, py, pz};
int periods[3] = {0, 0, 0};
MPI_Comm cart_comm;
MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

// Get coordinates of this process in process grid
int coords[3];
MPI_Cart_coords(cart_comm, rank, 3, coords);
int px_id = coords[0];
int py_id = coords[1];
int pz_id = coords[2];

MPI_Comm io_comm;
int color = (px_id == 0 && py_id == 0) ? 1 : MPI_UNDEFINED;
MPI_Comm_split(cart_comm, color, pz_id, &io_comm);

// Local block size (excluding halo)
int lx = nx / px;
int ly = ny / py;
int lz = nz / pz;

// Local dimensions including halos
int local_lx = lx + ((px_id > 0) ? 1 : 0) + ((px_id < px - 1) ? 1 : 0);
int local_ly = ly + ((py_id > 0) ? 1 : 0) + ((py_id < py - 1) ? 1 : 0);
int local_lz = lz + ((pz_id > 0) ? 1 : 0) + ((pz_id < pz - 1) ? 1 : 0);

// Allocate memory for all columns at once
double **local_data_all_cols = (double **)malloc(nc * sizeof(double *));
double **full_data_all_cols = (double **)malloc(nc * sizeof(double *));

for (int c = 0; c < nc; c++) {
    local_data_all_cols[c] = (double *)malloc(local_lx * local_ly * lz * sizeof(double));
    full_data_all_cols[c] = (double *)malloc(local_lx * local_ly * local_lz * sizeof(double));
    
    // Initialize to zero (optional)
    memset(local_data_all_cols[c], 0, local_lx * local_ly * lz * sizeof(double));
    memset(full_data_all_cols[c], 0, local_lx * local_ly * local_lz * sizeof(double));
}

if (px_id == 0 && py_id == 0) {
    // This is the designated reader in this layer
    double *slabcols = (double *)malloc(nx * ny * lz * nc * sizeof(double));
    
    MPI_File fh;
    MPI_File_open(io_comm, "./3data.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    int layer_size = nx * ny * lz * nc;
    MPI_Offset offset = (MPI_Offset)pz_id * layer_size * sizeof(double);

    // Read this layer
    MPI_File_read_at(fh, offset, slabcols, layer_size, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    
    // Reorganize data by column
    double **slab_by_col = (double **)malloc(nc * sizeof(double *));
    for (int c = 0; c < nc; c++) {
        slab_by_col[c] = (double *)malloc(nx * ny * lz * sizeof(double));
        
        // Extract each column from interleaved data
        for (int i = 0; i < nx * ny * lz; i++) {
            slab_by_col[c][i] = slabcols[c + i * nc];
        }
    }
    
    free(slabcols); // Free original interleaved data
    
    // Distribute all columns to all processes in this layer
    for (int i = 0; i < px; i++) {
        for (int j = 0; j < py; j++) {
            int dest_coords[3] = {i, j, pz_id};
            int dest_rank;
            MPI_Cart_rank(cart_comm, dest_coords, &dest_rank);
            
            int sx = lx * i + ((i) ? -1 : 0);
            int sy = ly * j + ((j) ? -1 : 0);
            int sizex = lx + ((i) ? 1 : 0) + ((i < px - 1) ? 1 : 0);
            int sizey = ly + ((j) ? 1 : 0) + ((j < py - 1) ? 1 : 0);
            
            int starts[3] = {0, sy, sx};
            int subsizes[3] = {lz, sizey, sizex};
            int bigsizes[3] = {lz, ny, nx};

            MPI_Datatype subarray;
            MPI_Type_create_subarray(3, bigsizes, subsizes, starts,
                                     MPI_ORDER_C, MPI_DOUBLE, &subarray);
            MPI_Type_commit(&subarray);
            
            for (int c = 0; c < nc; c++) {
                if (i == 0 && j == 0) {
                    // Copy to local buffer
                    for (int kk = 0; kk < lz; kk++)
                        for (int jj = 0; jj < sizey; jj++)
                            for (int ii = 0; ii < sizex; ii++) {
                                int slab_idx = (kk * ny * nx) + (jj * nx) + ii;
                                int local_idx = kk * sizey * sizex + jj * sizex + ii;
                                local_data_all_cols[c][local_idx] = slab_by_col[c][slab_idx];
                            }
                } else {
                    MPI_Send(slab_by_col[c], 1, subarray, dest_rank, c, MPI_COMM_WORLD);
                }
            }
            
            MPI_Type_free(&subarray);
        }
    }
    
    // Free slab data
    for (int c = 0; c < nc; c++) {
        free(slab_by_col[c]);
    }
    free(slab_by_col);
} else {
    // Non-reader processes receive their part of each column
    for (int c = 0; c < nc; c++) {
        MPI_Recv(local_data_all_cols[c], local_lx * local_ly * lz, MPI_DOUBLE, 
                MPI_ANY_SOURCE, c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// Determine neighbors in z-direction
int up_rank = MPI_PROC_NULL;  // Process above
int down_rank = MPI_PROC_NULL; // Process below

if (pz_id > 0) {
    int up_coords[3] = {px_id, py_id, pz_id - 1};
    MPI_Cart_rank(cart_comm, up_coords, &up_rank);
}

if (pz_id < pz - 1) {
    int down_coords[3] = {px_id, py_id, pz_id + 1};
    MPI_Cart_rank(cart_comm, down_coords, &down_rank);
}

// Create MPI datatype for the xy-plane (for z-direction communication)
MPI_Datatype xy_plane_type;
MPI_Type_vector(local_ly, local_lx, local_lx, MPI_DOUBLE, &xy_plane_type);
MPI_Type_commit(&xy_plane_type);

// Copy the interior data to the full data array for all columns
for (int c = 0; c < nc; c++) {
    // Initialize the full data array from local_data
    for (int k = 0; k < local_lz; k++) {
        for (int j = 0; j < local_ly; j++) {
            for (int i = 0; i < local_lx; i++) {
                // Initialize interior points from local_data
                // Skip the halo regions when initializing
                if ((k > 0 || pz_id == 0) && (k < local_lz - 1 || pz_id == pz - 1)) {
                    int local_k = (pz_id > 0) ? k - 1 : k;
                    int full_idx = k * local_ly * local_lx + j * local_lx + i;
                    int local_idx = local_k * local_ly * local_lx + j * local_lx + i;
                    full_data_all_cols[c][full_idx] = local_data_all_cols[c][local_idx];
                }
            }
        }
    }
    
    // Exchange data in z-direction for each column
    // Send bottom layer to process below and receive top halo from process above
    MPI_Sendrecv(
        full_data_all_cols[c] + (pz_id > 0 ? 1 : 0) * local_lx * local_ly, 1, xy_plane_type, down_rank, c,
        full_data_all_cols[c], 1, xy_plane_type, up_rank, c,
        cart_comm, MPI_STATUS_IGNORE
    );

    // Send top layer to process above and receive bottom halo from process below
    MPI_Sendrecv(
        full_data_all_cols[c] + (local_lz - (pz_id < pz - 1 ? 2 : 1)) * local_lx * local_ly, 1, xy_plane_type, up_rank, c + nc,
        full_data_all_cols[c] + (local_lz - 1) * local_lx * local_ly, 1, xy_plane_type, down_rank, c + nc,
        cart_comm, MPI_STATUS_IGNORE
    );
}

MPI_Type_free(&xy_plane_type);

// Make sure halo exchange is complete before starting computation
MPI_Barrier(cart_comm);

// Process offsets for interior points
int ox = (px_id > 0) ? 1 : 0;
int oy = (py_id > 0) ? 1 : 0;
int oz = (pz_id > 0) ? 1 : 0;

// Arrays to store results for all columns
double *local_minima = (double *)malloc(nc * sizeof(double));
double *local_maxima = (double *)malloc(nc * sizeof(double));
int (*local_count)[2] = (int (*)[2])malloc(nc * sizeof(int[2]));

// Initialize arrays
for (int c = 0; c < nc; c++) {
    local_minima[c] = 1e16;
    local_maxima[c] = -1e16;
    local_count[c][0] = 0; // minima count
    local_count[c][1] = 0; // maxima count
}

// Process all columns
for (int c = 0; c < nc; c++) {
    for (int k = oz; k < lz + oz; k++) {
        for (int j = oy; j < ly + oy; j++) {
            for (int i = ox; i < lx + ox; i++) {
                int idx = k * local_lx * local_ly + j * local_lx + i;
                int minima_flag = 1;
                int maxima_flag = 1;
                
                // Check left neighbor (x-direction)
                if (i > ox || px_id > 0) {
                    int idx_nbr = k * local_lx * local_ly + j * local_lx + (i-1);
                    if (full_data_all_cols[c][idx_nbr] <= full_data_all_cols[c][idx]) maxima_flag = 0;
                    if (full_data_all_cols[c][idx_nbr] >= full_data_all_cols[c][idx]) minima_flag = 0;
                }
                
                // Check right neighbor (x-direction)
                if (i < lx + ox - 1 || px_id < px - 1) {
                    int idx_nbr = k * local_lx * local_ly + j * local_lx + (i+1);
                    if (full_data_all_cols[c][idx_nbr] <= full_data_all_cols[c][idx]) maxima_flag = 0;
                    if (full_data_all_cols[c][idx_nbr] >= full_data_all_cols[c][idx]) minima_flag = 0;
                }
                
                // Check bottom neighbor (y-direction)
                if (j > oy || py_id > 0) {
                    int idx_nbr = k * local_lx * local_ly + (j-1) * local_lx + i;
                    if (full_data_all_cols[c][idx_nbr] <= full_data_all_cols[c][idx]) maxima_flag = 0;
                    if (full_data_all_cols[c][idx_nbr] >= full_data_all_cols[c][idx]) minima_flag = 0;
                }
                
                // Check top neighbor (y-direction)
                if (j < ly + oy - 1 || py_id < py - 1) {
                    int idx_nbr = k * local_lx * local_ly + (j+1) * local_lx + i;
                    if (full_data_all_cols[c][idx_nbr] <= full_data_all_cols[c][idx]) maxima_flag = 0;
                    if (full_data_all_cols[c][idx_nbr] >= full_data_all_cols[c][idx]) minima_flag = 0;
                }
                
                // Check front neighbor (z-direction)
                if (k > oz || pz_id > 0) {
                    int idx_nbr = (k-1) * local_lx * local_ly + j * local_lx + i;
                    if (full_data_all_cols[c][idx_nbr] <= full_data_all_cols[c][idx]) maxima_flag = 0;
                    if (full_data_all_cols[c][idx_nbr] >= full_data_all_cols[c][idx]) minima_flag = 0;
                }
                
                // Check back neighbor (z-direction)
                if (k < lz + oz - 1 || pz_id < pz - 1) {
                    int idx_nbr = (k+1) * local_lx * local_ly + j * local_lx + i;
                    if (full_data_all_cols[c][idx_nbr] <= full_data_all_cols[c][idx]) maxima_flag = 0;
                    if (full_data_all_cols[c][idx_nbr] >= full_data_all_cols[c][idx]) minima_flag = 0;
                }
                
                local_count[c][0] += minima_flag;
                local_count[c][1] += maxima_flag;
                
                if (full_data_all_cols[c][idx] > local_maxima[c]) local_maxima[c] = full_data_all_cols[c][idx];
                if (full_data_all_cols[c][idx] < local_minima[c]) local_minima[c] = full_data_all_cols[c][idx];
            }
        }
    }
}

// Arrays for global results
double *global_minima = NULL;
double *global_maxima = NULL;
int (*count)[2] = NULL;

if (rank == 0) {
    global_minima = (double *)malloc(nc * sizeof(double));
    global_maxima = (double *)malloc(nc * sizeof(double));
    count = (int (*)[2])malloc(nc * sizeof(int[2]));
}

// Reduce results for all columns
for (int c = 0; c < nc; c++) {
    double c_min, c_max;
    int c_count[2];
    
    MPI_Reduce(&local_minima[c], &c_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_maxima[c], &c_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_count[c], c_count, 2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        global_minima[c] = c_min;
        global_maxima[c] = c_max;
        count[c][0] = c_count[0];
        count[c][1] = c_count[1];
    }
}

// Print results for all columns
if (rank == 0) {
    for (int c = 0; c < nc; c++) {
        printf("Column %d: Min=%lf Max=%lf Minima_count=%d Maxima_count=%d\n", 
               c, global_minima[c], global_maxima[c], count[c][0], count[c][1]);
    }
    
    free(global_minima);
    free(global_maxima);
    free(count);
}

// Free memory
free(local_minima);
free(local_maxima);
free(local_count);

for (int c = 0; c < nc; c++) {
    free(local_data_all_cols[c]);
    free(full_data_all_cols[c]);
}
free(local_data_all_cols);
free(full_data_all_cols);

if (io_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&io_comm);
}
MPI_Comm_free(&cart_comm);
MPI_Finalize();

	return 0;
}