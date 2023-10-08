#include <stdio.h>
#include <complex.h>
#include <time.h>

#define Y 1080
#define X 1920

// Function to calculate the Mandelbrot set and write it to a PPM file
void MandelbrotToPPM() {
    clock_t start_time, end_time;
    double execution_time;
    start_time = clock(); // Record the start time
    FILE *ppmFile = fopen("mandelbrot.ppm", "w");
    
    if (!ppmFile) {
        perror("Error opening file");
        return;
    }
    
    fprintf(ppmFile, "P3\n%d %d\n255\n", X, Y);
    
    for (int y = 0; y < Y; y++) {
        for (int x = 0; x < X; x++) {
            double real = -2.0 + 3.0 * x / (X - 1);
            double imag = -1.0 + 2.0 * y / (Y - 1);
            double complex c = real + imag * I;
            
            double complex z = 0;
            int maxIter = 100;  // Maximum number of iterations
            int iter = 0;
            
            while (cabs(z) < 2 && iter < maxIter) {
                z = z * z + c;
                iter++;
            }
            
            // Map the number of iterations to a color
            int r = iter % 8 * 32;
            int g = iter % 16 * 16;
            int b = iter % 32 * 8;
            
            fprintf(ppmFile, "%d %d %d\n", r, g, b);
        }
    }
    
    fclose(ppmFile);
    
    end_time = clock(); // Record the end time
    execution_time = ((double)(end_time - start_time))/CLOCKS_PER_SEC;

    printf("Mandelbrot image saved as 'sequential.ppm'\n");
    printf("Execution time: %f seconds\n", execution_time);
    
}

int main() {
    MandelbrotToPPM();
    return 0;
}