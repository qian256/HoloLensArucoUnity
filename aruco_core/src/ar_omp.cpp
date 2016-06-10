#ifndef USE_OMP
int omp_get_max_threads() { return 1; }
int omp_get_thread_num() { return 0; }
int omp_set_num_threads(int){return 0;}

#endif
