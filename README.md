# HAC_introduction
HAC is a personal learning repository focused on exploring the world of hardware acceleration using tools like NVIDIA CUDA and High-Level Synthesis (HLS). It serves as a sandbox for experimenting with parallel computing, performance optimization, and low-level hardware interaction—from GPUs to FPGAs.

## good to know
- [Image convolution with cuda](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_64_website/projects/convolutionSeparable/doc/convolutionSeparable.pdf)



Nsight results:


Nsight Systems Analyse (Jetson Nano)
De performantieanalyse met Nsight Systems toont aan dat de grootste bottleneck zich bevindt in de functie cudaMemcpyToSymbol, goed voor 87% van de totale CUDA API-tijd (±518 ms voor 2 oproepen). Daarnaast veroorzaakt sem_timedwait 72% van de OS-runtime tijd, wat wijst op aanzienlijke CPU-wachttijden. De eigenlijke CUDA-kernels (row/column convoluties) zijn efficiënt met gemiddelde looptijden onder 3 ms.
🛠️ Optimalisatiesuggesties: vermijd blocking API-calls en vervang cudaMemcpyToSymbol door asynchrone geheugenoverdracht (cudaMemcpyAsync) waar mogelijk.

- [results(in png)](/results)


Disclaimer:

JetPack 4.6.1 does not include NVIDIA Nsight Compute  as part of the CUDA Toolkit 10.2 but JetPack 6.2 includes NVIDIA Nsight Compute v2023.2 as part of the CUDA Toolkit 12.2

https://developer.nvidia.com/embedded/jetpack

/opt/nvidia/nsight-compute lacks nsight compute


/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */
