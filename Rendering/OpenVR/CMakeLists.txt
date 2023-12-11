vtk_module_find_package(PACKAGE OpenVR)

set(classes
  vtkOpenVRCamera
  vtkOpenVRControlsHelper
  vtkOpenVRDefaultOverlay
  vtkOpenVRInteractorStyle
  vtkOpenVRModel
  vtkOpenVROverlay
  vtkOpenVRRenderer
  vtkOpenVRRenderWindow
  vtkOpenVRRenderWindowInteractor)

vtk_encode_string(
  INPUT         "OpenVRDashboard.jpg"
  HEADER_OUTPUT dashboard_header
  SOURCE_OUTPUT dashboard_source
  BINARY)

vtk_module_add_module(VTK::RenderingOpenVR
  CLASSES ${classes}
  SOURCES "${dashboard_source}" ${vtk_object_factory_source} ${geometry_sources}
  PRIVATE_HEADERS "${dashboard_header}" ${vtk_object_factory_header} ${geometry_headers})
vtk_module_link(VTK::RenderingOpenVR
  PUBLIC
    OpenVR::OpenVR)

set(openvr_input_files
  vtk_openvr_actions.json
  vtk_openvr_binding_hpmotioncontroller.json
  vtk_openvr_binding_knuckles.json
  vtk_openvr_binding_oculus_touch.json
  vtk_openvr_binding_vive_controller.json
)

foreach(inputfile IN LISTS openvr_input_files)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/${inputfile}"
    "${CMAKE_BINARY_DIR}/${input_file}"
    COPYONLY)
endforeach()

install(
  FILES ${openvr_input_files}
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT   runtime
)

# testing
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/ThirdParty/mint/interop interop_build)
if (TARGET interop)
    if (TARGET RenderingOpenVR)
    message("adding interop")
    target_link_libraries(RenderingOpenVR PRIVATE interop)
    set_property(TARGET RenderingOpenVR PROPERTY CXX_STANDARD 17)
    endif()
endif()


# GLM
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE OFF CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_CXX_14 ON CACHE BOOL "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/ThirdParty/mint/rendering/external/glm/ glm_build)
if (TARGET glm)
    if (TARGET RenderingOpenVR)
    message("adding glm")
    target_link_libraries(RenderingOpenVR PRIVATE glm)
    set_property(TARGET RenderingOpenVR PROPERTY CXX_STANDARD 17)
    endif()
endif()