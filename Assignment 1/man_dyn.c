#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <time.h>


#define MAX_WIDTH 10000
#define MAX_HEIGHT 10000
#define MASTER 0
#define TAG 0
#define height 600
#define width 800

typedef struct {
    float real;
    float imag;
} Complex;

int cal_pixel(Complex c) {
    Complex z;
    int count, max;
    float temp, lengthsq;
    max = 256;
    z.real = 0;
    z.imag = 0;
    count = 0;

    do {
        temp = z.real * z.real - z.imag * z.imag + c.real;
        z.imag = 2 * z.real * z.imag + c.imag;
        z.real = temp;
        lengthsq = z.real * z.real + z.imag * z.imag;
        count++;
    } while ((lengthsq < 4.0) && (count < max));

    return count;
}

bool inBounds(int w, int h) {
    if ((w < 0) || (w > MAX_WIDTH) || (h < 0) || (h > MAX_HEIGHT)) {
        printf("ERROR: Invalid image dimensions.\n");
        printf("Height and width range: [1, 32000]\n");
        return false;
    }
    return true;
}

void outputResults(unsigned char **pixs, double time, int w, int h) {
    const unsigned char **p;
    p = (const unsigned char **)pixs;
    printf("Execution time= %.6f s.\n", time);

    // Write image to a PPM file
    FILE *ppmFile = fopen("dynamic.ppm", "wb");
    fprintf(ppmFile, "P6\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            fputc(p[y][x], ppmFile);  // Red
            fputc(p[y][x], ppmFile);  // Green
            fputc(p[y][x], ppmFile);  // Blue
        }
    }
    fclose(ppmFile);
    printf("Mandelbrot image saved as 'dynamic.ppm'\n");
}

int main(int argc, char *argv[]) {
    int comm_sz, my_rank;
    int kill, proc, count, rcvdRow;
    int row, col, greyScaleMod;
    double radius, radSq;
    double deltaTime;
    Complex c;
    unsigned char *pixels;
    bool go;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Assign default parameters
    radius = 2.0;
    radSq = (radius * radius);
    row = 0;
    count = 0;
    kill = -1;
    greyScaleMod = 256;

    // Test for correct user input
    if (!inBounds(width, height)) {
        MPI_Finalize();
        return 0;
    }

    // 1D array to hold single row data
    pixels = (unsigned char *)calloc(width, sizeof(unsigned char));

    // Parallelization portion of Mandelbrot set algorithm
    if (my_rank == MASTER) {
        // 2D array to store entire image data
        unsigned char **p = (unsigned char **)malloc(height * sizeof(unsigned char *));
        for (int index = 0; index < height; index++) {
            p[index] = (unsigned char *)malloc(width * sizeof(unsigned char));
        }

        // start the wall clock
        printf("width x height = %d x %d\n", width, height);
        clock_t startTime = clock();

        // Give all processors one row each to start
        proc = 1;
        while (proc < comm_sz) {
            MPI_Send(&row, 1, MPI_INT, proc, TAG, MPI_COMM_WORLD);
            proc++;
            row++;
            count++;
        }

        // Give out one row at a time as processors become available
        MPI_Status status;
        do {
            MPI_Recv(pixels, width, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            proc = status.MPI_SOURCE;
            rcvdRow = status.MPI_TAG;
            count--;

            if (row < height) {
                MPI_Send(&row, 1, MPI_INT, proc, TAG, MPI_COMM_WORLD);
                row++;
                count++;
            } else {
                MPI_Send(&kill, 1, MPI_INT, proc, TAG, MPI_COMM_WORLD);
            }

            // Store received row in 2D image array
            for (col = 0; col < width; col++) {
                p[rcvdRow][col] = pixels[col];
            }
        } while (count > 0);

        // Stop timer, store elapsed time, and report to user
        clock_t endTime = clock();
        deltaTime = (double)(endTime - startTime)/CLOCKS_PER_SEC;

        // Report elapsed time and write image to file
        outputResults(p, deltaTime, width, height);

        // Deallocate 2D array
        for (int index = 0; index < height; index++) {
            free(p[index]);
        }
        free(p);
    } else { /* Slaves process one row of pixels sent from MASTER and send back */
        go = true;
        while (go) {
            MPI_Recv(&row, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (row == -1) {
                go = false;
            } else {
                for (col = 0; col < width; col++) {
                    c.real = (col - width / radius) * (radSq / width);
                    c.imag = (row - height / radius) * (radSq / width);

                    // Individual pixel stored as 8-bit color value (0-255)
                    pixels[col] = (cal_pixel(c) * 35) % greyScaleMod;
                }
                MPI_Send(pixels, width, MPI_UNSIGNED_CHAR, MASTER, row, MPI_COMM_WORLD);
            }
        }
    }

    // Deallocate, wait, and shutdown
    free(pixels);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

