SET(TARGET_SRC bench.cc)

ADD_EXECUTABLE( "benchmark_precond" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("benchmark_precond")
set_target_properties("benchmark_precond" PROPERTIES OUTPUT_NAME "bench")
TARGET_LINK_LIBRARIES("benchmark_precond" "data_locality")
