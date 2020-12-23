SET(TARGET_SRC bench.cc)

ADD_EXECUTABLE( "benchmark_precond_merged" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("benchmark_precond_merged")
set_target_properties("benchmark_precond_merged" PROPERTIES OUTPUT_NAME "bench")
TARGET_LINK_LIBRARIES("benchmark_precond_merged" "data_locality")
