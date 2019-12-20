
#include "histdupe.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#include <chrono>



/*
Utility method for launching a CUDA kernel. Performes all the nasty checks, allocation, and copying of data.
*/
cudaError_t findDupes(const float*, const unsigned int, std::vector<Pair>&, unsigned int*, const unsigned int, const float*);



int main(int argc, char* argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Too few arguments. Expected 1\n\nUsage: %s DATA_PATH\n", argv[0]);
        return 1;
    }


    // Initialize variables
    // Maximum number of results to return. Only applies to CUDA launches
    unsigned int max_results = 1000000;
    // Base confidence value for similar pairs
    float confidence = 0.95f;
    // Maximum color variance per histogram. Used to determine if an image is black-and-white or colorful
    float color_variance = 0.25f;
    // Number of histograms in the dataset
    unsigned int N = 50000;
    // Use CUDA to find similar pair
    bool cuda = true;

    // Clock used for timing
    std::chrono::steady_clock::time_point time;


    // Print some diagnostics
    std::cout << "Datafile Path: " << argv[1] << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "Max Results: " << max_results << std::endl;
    std::cout << "Confidence: " << confidence << std::endl;
    std::cout << "Color Variance: " << color_variance << std::endl;


    // Allocate some arrays
    std::cout << "Allocating memory..." << std::endl;
    time = std::chrono::steady_clock::now();

    int* ids = new int[N]; // Mapping of actual index to ID of histogram
    float* data = new float[128 * N]; // End-to-end array of all histograms. Each histogram consists of 128 floats
    float* conf = new float[N]; // Confidence array; allows using stricter confidence for black and white images
    std::vector<Pair> pairs; // Vector of similar pairs (to be populated)

    std::cout << "Allocated memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;



    // Read test data from file
    std::cout << "Reading data from file: " << argv[1] << "..." << std::endl;
    time = std::chrono::steady_clock::now();

    FILE* file; // Data file
    file = fopen(argv[1], "r"); // Open data file to read
    for (unsigned int i = 0; i < N; i++) {
        fscanf(file, "%d", &ids[i]); // Read first int as id of histogram
        for (int j = 0; j < 128; j++) { // Read 128 floats as histogram elements
            fscanf(file, "%f", &data[i * 128 + j]);
        }
    }
    fclose(file); // Close data file

    std::cout << "Read data in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;



    // Build confidence array
    std::cout << "Building confidence array..." << std::endl;
    time = std::chrono::steady_clock::now();

    float confidence_square = 1 - (1 - confidence) * (1 - confidence); // Squared confidence for comparing black and white images.
    // Generate confidence array
    for (unsigned int i = 0; i < N; i++) {
        float d = 0;

        // Compute sum of color variance across histogram
        for (unsigned int k = 0; k < 32; k++) {
            // Ignore alpha values (first 32 floats)
            float r = data[i * 128 + k + 32];
            float g = data[i * 128 + k + 64];
            float b = data[i * 128 + k + 96];
            d += __max(__max(r, g), b) - __min(__min(r, g), b);
        }

        if (d > color_variance) {
            conf[i] = confidence; // Image is colorful, use normal confidence
        } else {
            conf[i] = confidence_square; // Image is not colorful, use squared confidence
        }
    }

    std::cout << "Built confidence array in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


    // Find duplicates
    std::cout << "Finding duplicates..." << std::endl;
    time = std::chrono::steady_clock::now();

    cudaError_t cudaStatus; // CUDA Status variable
    unsigned int result_count = 0; // Track number of results
    if (cuda) {
        // With CUDA
        cudaStatus = findDupes(data, N, pairs, &result_count, max_results, conf);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel failed!");
            return 1;
        }

        // Convert indexes into histogram ids
        for (unsigned int i = 0; i < result_count; i++) {
            pairs[i].id1 = ids[pairs[i].id1];
            pairs[i].id2 = ids[pairs[i].id2];
        }
    } else {
        // Sequentially
        for (unsigned int i = 0; i < N; i++) {
            for (unsigned int j = i + 1; j < N; j++) {
                double d = 0;
                for (unsigned int k = 0; k < 128; k++) {
                    d += fabs(data[i * 128 + k] - data[j * 128 + k]);
                }
                d = 1 - (d / 8);
                if (d > fmaxf(conf[i], conf[j])) { // Use highest confidence value of the two histograms
                    Pair p;
                    p.similarity = (float) d;
                    p.id1 = ids[i];
                    p.id2 = ids[j];
                    pairs.push_back(p);
                    result_count++;
                }
            }
        }
    }

    std::cout << "Found duplicates in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


    // Print some results
    std::cout << "Found pairs: " << result_count << std::endl;
    std::cout << "Example results:" << std::endl;
    for (unsigned int i = 0; i < __min(result_count, 10); i++) {
        std::cout << "\t" << ids[pairs[i].id1] << " - " << ids[pairs[i].id2] << ":\t\t" << pairs[i].similarity << std::endl;
    }


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    // Delete arrays
    delete[] data;
    delete[] conf;
    delete[] ids;

    return 0;
}



cudaError_t findDupes(const float* data, const unsigned int N, std::vector<Pair>& pairs, unsigned int* result_count, const unsigned int max_results, const float* confidence) {

    float* d_data; // Data device pointer
    Pair* d_pairs; // Pairs device pointer
    float* d_confidence; // Confidence device pointer
    unsigned int* d_result_count; // Result count device pointer
    cudaError_t cudaStatus; // CUDA error

    std::chrono::steady_clock::time_point time; // Time tracking


    unsigned int dN = N; // Padded device N to match block size
    if (N % 64 != 0) {
        dN = (unsigned int) ceil((double) N / 64) * 64;
    }
    std::cout << "Adjusted N: " << dN << std::endl;


    // Choose which GPU to run on, change this on a multi-GPU system
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }


    // Allocate GPU buffers
    std::cout << "Allocating GPU memory..." << std::endl;
    time = std::chrono::steady_clock::now();
    cudaStatus = cudaMalloc((void**) &d_data, sizeof(float) * 128 * dN); // Allocate memory for histogram data
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**) &d_pairs, sizeof(Pair) * max_results); // Allocate memory for results
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**) &d_result_count, sizeof(unsigned int)); // Allocate single int for result count
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**) &d_confidence, sizeof(float) * dN); // Allocate memory for confidence array
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    std::cout << "Allocated GPU memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


    // Copy input data from host memory to GPU buffers
    std::cout << "Copying data to device..." << std::endl;
    time = std::chrono::steady_clock::now();
    cudaStatus = cudaMemcpy(d_data, data, sizeof(int) * 128 * N, cudaMemcpyHostToDevice); // Copy histogram data to device
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_confidence, confidence, sizeof(float) * N, cudaMemcpyHostToDevice); // Copy confidence array to device
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    if (dN > N) {
        // Copy padded data to device at end of confidence array
        float* temp_conf = new float[dN - N]; // Temp array of padded confidence values
        for (unsigned int i = 0; i < dN - N; i++) temp_conf[i] = 2; // Impossible confidence
        cudaStatus = cudaMemcpy(d_confidence + N, temp_conf, sizeof(float) * (dN - N), cudaMemcpyHostToDevice); // Copy padded confidence values to device
        delete[] temp_conf; // Delete temp array
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    }
    std::cout << "Copied data to GPU memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


    // Launch a kernel on the GPU
    std::cout << "Launching kernel..." << std::endl;
    time = std::chrono::steady_clock::now();
    histDupeKernel KERNEL_ARGS((int) ceil((double) N / 64), 64) (d_data, d_confidence, d_pairs, d_result_count, N, max_results); // Launch CUDA kernel


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }
    std::cout << "Ran GPU kernel in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


    // Copy output from GPU buffer to host memory.
    std::cout << "Copying results from device..." << std::endl;
    time = std::chrono::steady_clock::now();
    cudaStatus = cudaMemcpy((void*) result_count, d_result_count, sizeof(float), cudaMemcpyDeviceToHost); // Copy result count from device
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    result_count[0] = __min(result_count[0], max_results); // Clamp result_count to max_results
    // Read result pairs into buffer
    {
        Pair* temp_pairs = new Pair[result_count[0]]; // Result set pair buffer
        cudaStatus = cudaMemcpy((void*)temp_pairs, d_pairs, sizeof(Pair) * result_count[0], cudaMemcpyDeviceToHost); // Copy results from device
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
        // Only keep pairs that are unique (pairs are commutative)
        for (unsigned int i = 0; i < result_count[0]; i++) {
            int p1id1 = temp_pairs[i].id1;
            int p1id2 = temp_pairs[i].id2;
            bool found = false;
            for (const Pair p2 : pairs) {
                if ((p1id1 == p2.id1 && p1id2 == p2.id2) || (p1id1 == p2.id2 && p1id2 == p2.id1)) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                pairs.push_back(temp_pairs[i]); // Only keep pair if it is unique
            }
        }
        delete[] temp_pairs; // Delete temp results buffer
    }
    result_count[0] = (int) pairs.size(); // Reset result_count to count of final result set
    std::cout << "Retrieved results from GPU memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


Error:
    // Free cuda memory
    std::cout << "Freeing GPU memory..." << std::endl;
    time = std::chrono::steady_clock::now();
    cudaFree(d_data);
    cudaFree(d_pairs);
    cudaFree(d_result_count);
    cudaFree(d_confidence);
    std::cout << "Freed GPU memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;

    return cudaStatus;
}
