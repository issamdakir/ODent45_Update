import os
from os.path import dirname,basename, join, abspath, expanduser,exists, isfile, isdir
from glob import glob
import zipfile
from importlib import import_module
import socket
import webbrowser
import sys
import json
import shutil
import requests
import tempfile
from requests.exceptions import HTTPError, Timeout, RequestException
import bpy  # type: ignore
from bpy.props import ( # type: ignore
    StringProperty,
    
)

import gpu # type: ignore
from gpu_extras.batch import batch_for_shader # type: ignore
import blf # type: ignore
from time import sleep, perf_counter
import subprocess
import platform
import weakref


def get_odent_version(filepath=None):
    if filepath is None :
        return "####"

    try :
        with open(filepath, "r") as rf:
            lines = rf.readlines()
            version = int(lines[0])
            return version
    except Exception as er :
        txt_message = [f""]
        odent_log(txt_message)
        return "####"
    
def update_is_availible() :
    if OdentConstants.ADDON_VER_DATE == "####" or not isConnected():
        return None
    remote_version,success,error_txt_list = get_update_version()
            
    if not success or remote_version <= OdentConstants.ADDON_VER_DATE :
        return None
    
    return remote_version

class AreaTagManager:
    def __init__(self):
        self._data = weakref.WeakKeyDictionary()

    def tag(self, area, key, value):
        """Assign a value to a custom key for a given area."""
        if area not in self._data:
            self._data[area] = {}
        self._data[area][key] = value

    def get(self, area, key, default=None):
        """Retrieve the value of a custom key for a given area."""
        return self._data.get(area, {}).get(key, default)

    def has(self, area, key):
        """Check if a key exists for the given area."""
        return key in self._data.get(area, {})

    def remove(self, area, key):
        """Remove a key for the given area, if it exists."""
        if area in self._data and key in self._data[area]:
            del self._data[area][key]

    def clear(self, area):
        """Remove all tags for a given area."""
        if area in self._data:
            del self._data[area]

class OdentConstants():
    """Odent constants"""

    ODENT_LIB_NAME='Odent_Library'
    ODENT_LIB_ARCHIVE_NAME='Odent_Library_Archive'
    ADDON_VER_NAME = "ODENT_Version.txt"
    ODENT_MODULES_NAME = "odent_modules"
    CONFIG_ZIP_NAME = "config.zip"
    STARTUP_FILE_NAME = "startup.blend"

    ADDON_NAME = "Odent45"
    ADDON_RELEASE = "beta"
    
    ADDON_DIR = dirname(abspath(__file__))
    

    BLENDER_ROOT_PATH = dirname(dirname(dirname(ADDON_DIR)))
    RESOURCES = join(ADDON_DIR, "Resources")
    ADDON_VER_PATH = join(RESOURCES, ADDON_VER_NAME)
    ADDON_VER_DATE = get_odent_version(filepath=ADDON_VER_PATH)
    CONFIG_ZIP_PATH = join(RESOURCES, CONFIG_ZIP_NAME)
    STARTUP_FILE_PATH = join(RESOURCES, STARTUP_FILE_NAME)
    ODENT_LIBRARY_PATH = join(ADDON_DIR, ODENT_LIB_NAME)
    ODENT_LIBRARY_ARCHIVE_PATH = join(RESOURCES, ODENT_LIB_ARCHIVE_NAME)
    
    ODENT_MODULES_PATH = join(ADDON_DIR, ODENT_MODULES_NAME)
    MESH_REG_AUTO_PATH = join(ADDON_DIR, "operators","MeshRegAuto.exe")
    DATA_BLEND_FILE = join(RESOURCES, "BlendData","ODENT_BlendData.blend")
    ODENT_APP_TEMPLATE_PATH = join(RESOURCES, "odent_app_template.zip")

    BLF_INFO = {
    "fontid" : 0,
    "size" : 14,
    }   
    REPO_URL = "https://github.com/issamdakir/Odent45/zipball/main"
    VERSION_URL = "https://raw.githubusercontent.com/issamdakir/Odent45/main/Resources/ODENT_Version.txt"
    ADDON_UPDATE_URL = "https://github.com/issamdakir/Odent45_Update/zipball/main"
    ADDON_UPDATE_NAME = "Odent_45_update"
    UPDATE_MAP_JSON = "update_data_map.json"
    UPDATE_VERSION_URL = "https://raw.githubusercontent.com/issamdakir/Odent45_Update/main/data/ODENT_Version.txt"
    TELEGRAM_LINK = "https://t.me/bdental3"
    REQ_DICT = {
        "SimpleITK": "SimpleITK",
        "vtk": "vtk",
        "cv2.aruco": "opencv-contrib-python",
        # "itk": "itk",
    }
    
    
    

    MAIN_WORKSPACE_NAME = "Odent Main"
    SLICER_WORKSPACE_NAME = "Odent Slicer"

    VISUALISATION_MODE_PCD = "Point cloud 3D"
    VISUALISATION_MODE_TEXTURED = "Textured 3D Vizualisation"
    
    VOXEL_OBJECT_NAME = "Voxel_Vizualization"
    VOXEL_OBJECT_TYPE = "Voxel_Vizualization_Object"
    DICOM_VIZ_COLLECTION_NAME = "Visualization_3D_Collection"

    VOXEL_PLANE_NAME = "Voxel_Plane"
    VOXEL_PLANE_TYPE = "Voxel_Plane_Object"
    VOXEL_PLANE_MAT_NAME = "Voxel_Plane_Mat"

    VOXEL_IMAGE_NAME = "Voxel_Image"
    VOXEL_IMAGE_TYPE = "Voxel_Image_Object"

    VOXEL_GROUPNODE_NAME = "Voxel_GroupNode"
    

    SLICES_POINTER_NAME = "Slices_Pointer"
    SLICES_POINTER_TYPE = "slices_pointer"
    SLICES_POINTER_COLLECTION_NAME = "Slices_Pointer_Collection"

    SLICES_SHADER_NAME = 'VGS_Dakir_Slices'

    SLICE_PLANE_TYPE = "slice_plane"
    SLICES_COLLECTION_NAME = "Slices_collection"

    AXIAL_SLICE_NAME = "Axial_Slice"
    SAGITTAL_SLICE_NAME = "Sagittal_Slice"
    CORONAL_SLICE_NAME = "Coronal_Slice"
    SLICE_CAM_TYPE ="slice_camera"

    SLICE_IMAGE_TYPE = "slice_image"

    DICOM_MESH_NAME = "DICOM_Mesh"
    DICOM_MESH_TYPE = "dicom_mesh_object"

    # "VGS_Marcos_modified_MinMax"#"VGS_Marcos_modified"  # "VGS_Marcos_01" "VGS_Dakir_01"
    ODENT_VOXEL_SHADER = "VGS_Dakir_(-400-3000)"#"VGS_Dakir_01"#"VGS_Dakir_MinMax"
    GUIDE_COMPONENTS_COLLECTION_NAME = "Surgical_Guide_Components_Collection"
    BOOL_NODE ="boolean_geonode"
    GUIDE_NAME = "Odent_guide"
    ODENT_IMAGE_METADATA_TAG = "odent_image"
    ODENT_IMAGE_METADATA_KEY = "image_type"
    ODENT_IMAGE_METADATA_UID_KEY = "uid"
    WMIN = -400
    WMAX = 3000
    BONE_UINT_THRESHOLD = 100
    BONE_FLOAT_THRESHOLD = 0.0
    XRAY_ALPHA = 0.9
    CAM_CLIP_OFFSET = 1
    CAM_DISTANCE = 100
    COLOR_CURVE_MAP = {1: (0.351, 0.265), 2: (0.721, 0.860)}
    ODENT_VOLUME_NODE_NAME = "odent_volume"
    VIEW_MATRIX = (
                (0.9205, -0.3908, 0.0, -25),
                (0.2772,  0.6528, 0.7050, -90),
                (-0.2755, -0.6489, 0.7092, -25),
                (0.0000, 0.0000, 0.0000, 1.0000),
            )
    
    # Selected icons :
    RED_ICON = "COLORSET_01_VEC"
    ORANGE_ICON = "COLORSET_02_VEC"
    GREEN_ICON = "COLORSET_03_VEC"
    BLUE_ICON = "COLORSET_04_VEC"
    VIOLET_ICON = "COLORSET_06_VEC"
    YELLOW_ICON = "COLORSET_09_VEC"
    YELLOW_POINT = "KEYTYPE_KEYFRAME_VEC"
    BLUE_POINT = "KEYTYPE_BREAKDOWN_VEC"
    SLICE_SEGMENT_COLOR_RGB = [0.68, 0.8, 1.0, 1.0]#[0.62, 1.0, 0.22, 1.0]
    SLICE_SEGMENT_COLOR_RATIO = 0.35

    ODENT_TEXT_3D_TYPE = "odent_text_3d"
    ODENT_TEXT_3D_MAT_NAME = "odent_text_3d_mat"
    ODENT_TEXT_3D_MAT_DIFFUSE_COLOR = [0, 0, 1, 1]


    ODENT_IMPLANT_TYPE = "odent_implant"
    ODENT_IMPLANT_NAME_PREFFIX = "Odent_Implant"
    ODENT_IMPLANT_COLLECTION_NAME = "Implant_Collection"
    IMPLANT_SAFE_ZONE_TYPE = "odent_implant_safe_zone"
    IMPLANT_SLEEVE_TYPE = "odent_implant_sleeve"
    IMPLANT_PIN_TYPE = "odent_implant_pin"
    FIXING_PIN_TYPE = "odent_fixing_pin"
    FIXING_SLEEVE_TYPE = "odent_fixing_sleeve"

    MHA_PATH_TAG = "mha_path"
    SOURCE_MHA_TAG = "source_mha"
    ODENT_TYPE_TAG = "odent_type"
    IMPLANT_LOCKED_MAT_NAME = "implant_locked_mat"
    IMPLANT_MAT_TAG = "implant_mat"
    ODENT_IMPLANT_REMOVE_CODE_TAG = "odent_remove_code"
    VOXEL_NODE_NAME_TAG = "voxel_node_name"

    PCD_OBJECT_NAME = "Point_Cloud_Vizualization"
    PCD_OBJECT_TYPE = "Point_Cloud_Vizualization_Object"
    PCD_MAT_NAME = "pcd_mat"
    PCD_INTENSITY_ATTRIBUTE_NAME = "voxel_intensity"
    PCD_THRESHOLD_NODE_NAME = "pcd_threshold"
    PCD_OPACITY_NODE_NAME = "pcd_opacity"
    PCD_POINT_RADIUS_NODE_NAME = "pcd_point_radius"
    PCD_POINT_AUTO_RESIZE_NODE_NAME = "pcd_point_auto_resize"
    PCD_POINT_EMISSION_NODE_NAME = "pcd_emission"
    PCD_GEONODE_NAME = "pcd_geonode"
    PCD_GEONODE_NAME_TAG = "pcd_geonode_name"
    PCD_GEONODE_MODIFIER_NAME = "pcd_geonode_modifier"
    PCD_MAX_POINTS = 6 # millions
    PCD_SAMPLING_METHOD_GRID = "Grid sampling"
    PCD_SAMPLING_METHOD_RANDOM = "Random sampling"

    CUTTERS_COLL_NAME = "Odent Cutters"
    CONNECT_PATH_CUTTER_NAME = "Connected path cutter"
    CONNECT_PATH_CUTTER_TYPE = "connected_path_cutter"
    CONNECT_PATH_CUTTER_MAT = {
        "name" : "connected_path_cutter_mat",
        "diffuse_color" : [0.1, 0.4, 0.7, 1.0],
        "roughness" : 0.3
        }


    LOCKED_TO_POINTER_MAT_NAME = "odent_locked_to_pointer_mat"
    PREVIOUS_ACTIVE_MAT_NAME_TAG = "odent_previous_mat"

    SPLIT_CUTTER_HOOK_POINT = "split_cutter_hook_point"
    CURVE_CUTTER1_TAG = "curvecutter1"
    CURVE_CUTTER2_TAG = "curvecutter2"

    ODENT_TEMP_DIR = join(expanduser("~"), ".odent_temp")
    ODENT_SLICES_DIR = join(ODENT_TEMP_DIR, "slices")
    ODENT_VOLUME_TEXTURES_DIR = join(ODENT_TEMP_DIR, "volume_textures")

    SPLINT_MAT_NAME = "mat_odent_splint"
    SPLINT_COLOR_DARK_GREEN = [0.0, 0.23, 0.2, 1.0]
    SPLINT_COLOR_BLUE = [0.0, 0.23, 0.2, 1.0]

class TimerLogger:
    def __init__(self, label=""):
        self.label = label
        self.start_time = perf_counter()
        self.last_time = self.start_time
        print(f"[{self.label}] Start")

    def log(self, step_name=""):
        now = perf_counter()
        step_elapsed = now - self.last_time
        total_elapsed = now - self.start_time

        print(f"[{self.label}] Step: {step_name} | +{step_elapsed:.6f}s | Total: {total_elapsed:.6f}s")
        self.last_time = now

    def end(self):
        now = perf_counter()
        total_elapsed = now - self.start_time
        print(f"[{self.label}] End | Total elapsed: {total_elapsed:.6f}s")

class ODENT_OT_MessageBox(bpy.types.Operator):
    """Odent popup message"""

    bl_idname = "wm.odent_message_box"
    bl_label = "ODENT INFO"
    bl_options = {"REGISTER"}

    message: StringProperty() # type: ignore
    icon: StringProperty() # type: ignore

    def execute(self, context):
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.alert = True
        box.alignment = "EXPAND"
        message = eval(self.message)
        for txt in message:
            row = box.row()
            row.label(text=txt)

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=300)

class ODENT_OT_OdentModulesPipInstaller(bpy.types.Operator):
    """pip install odent modules"""

    bl_idname = "wm.odent_modules_pip_installer"
    bl_label = "Install Odent Modules Online"
    bl_options = {"REGISTER"}

    

    def execute(self, context):
        if not isConnected(debug=True):
            message = ["No internet connection!"]
            odent_log(message)
            ODENT_GpuDrawText(message_list=message,rect_color=OdentColors.red,sleep_time=2)
            return {"CANCELLED"}
        target_dir_path = OdentConstants.ODENT_MODULES_PATH
        message = pip_install_modules(self.modules_list, target_dir_path)
        
        if message:
            odent_log(message)
            ODENT_GpuDrawText(message_list=message,rect_color=OdentColors.red,sleep_time=2)
            return {"CANCELLED"}
        else:
            message = ["Modules installed successfully, Please restart Blender."]
            odent_log(message)
            ODENT_GpuDrawText(message_list=message,rect_color=OdentColors.green)
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.alert = True
        box.alignment = "EXPAND"
        message = ["Connection to internet is required to install the modules.",
                    "Please make sure you are connected to the internet.",
                    "Click OK to continue or Right-click to abort."]
        g = box.grid_flow(columns=1, align=True)
        for txt in message:
            g.label(text=txt)
        g.label(text="Modules to install :")
        for txt in self.modules_list:
            g.label(text=txt)
        

    def invoke(self, context, event):
        self.modules_list = import_required_modules()
        if not self.modules_list:
            message = ["Odent modules already installed!"]
            odent_log(message)
            ODENT_GpuDrawText(message_list=message,rect_color=OdentColors.green,sleep_time=2)
            return {"CANCELLED"}
        
        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=800)


###########################################

DRAW_HANDLERS=[]

#Utils :
####################################################################################################

def clear_terminal():
    os.system("cls") if os.name == "nt" else os.system("clear")


def AbsPath(P):
    if P.startswith("//"):
        P = abspath(bpy.path.abspath(P))
    return P


def RelPath(P):
    if not P.startswith("//"):
        P = bpy.path.relpath(abspath(P))
    return P
           
def pip_install_modules(modules_list, target_dir_path=None):
    """Install required modules using pip"""
    error_message = []

    info_text = f"Updating pip ..."
    ODENT_GpuDrawText(message_list=[info_text])
    try :
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pip"])
    except subprocess.CalledProcessError as e:
        info_text = f"Updating pip failed!"
        odent_log([f"Updating pip failed! : {e}"])
        
    if target_dir_path is not None and not exists(target_dir_path):
        os.makedirs(target_dir_path)
    for module in modules_list:
        info_text = f"Installing {module} ..."
        ODENT_GpuDrawText(message_list=[info_text])
        try:
            if target_dir_path is not None:  
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", module, "--target", target_dir_path])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", module])
        except subprocess.CalledProcessError as e:
            info_text = f"Installing {module} failled!"
            ODENT_GpuDrawText(message_list=[info_text],rect_color=OdentColors.red)
            error_message.append(module)
    return error_message

def get_incremental_idx(data =None,odent_type=None):
    
    idx = 0
    if data is not None :
        objects = data
        if odent_type :
            objects = [obj for obj in data if obj.get("odent_type") == odent_type]
            if objects :
                idx = len(objects) + 1
    return idx
def get_incremental_name( name_preffix="", idx1=None, idx2=None, idx3=None):
    
    name = f"{name_preffix}_{idx1}"
    for idx in [idx2, idx3]:
        if idx is None:
            break
        name += f"_{idx}"
    return name

def is_linux():
    return platform.system() == "Linux"

def is_wine_installed():
    return shutil.which("wine") is not None

def install_wine():
    message = []
    try:
        print("Installing Wine...")
        subprocess.run(["sudo", "apt", "update"], check=True)
        subprocess.run(["sudo", "apt", "install", "-y", "wine"], check=True)
        subprocess.run(["sudo", "apt", "install", "-y", "wine64"], check=True)
        print("Wine installed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error during Wine installation:", e)
        message.append("Wine installation failed.")
    return message

def run_exe_with_wine(exe_path):
    if not os.path.exists(exe_path):
        print(f"File not found: {exe_path}")
        return
    try:
        subprocess.run(["wine", exe_path], check=True)
    except subprocess.CalledProcessError as e:
        print("Error running the exe file with Wine:", e)
        
def odent_log(txt_list,header=None,footer=None):
    _header, _footer = header, footer
    if _header is None :
        _header=f"\n{'#'*20} Odent log :  {'#'*20}\n"
    if _footer is None:
        _footer=f"\n{'#'*20} End log.\  {'#'*20}\n"
    
    print(_header)
    for line in txt_list :
        print(line)
    print(_footer)


def get_update_version(filepath=OdentConstants.UPDATE_VERSION_URL)  :
    success = False
    txt_list= []
    update_version = None
    try :
        r = requests.get(filepath)
        success = r.ok
        if not success :
            odent_log([f"connection to github update server failed! :\n{r.text}"])
            txt_list.append("connection to update server failed!")
        else :
            update_version = int(r.text)

    except Exception as er :
        txt_list.append("connection to update server failed!")
        odent_log([f"connection to github update server failed! :\n{er}"])
        success = False
        
    return update_version,success,txt_list
    
# class OdentColors():
#     white = [0.8,0.8,0.8,1.0]
#     black = [0.0,0.0,0.0,1.0]
#     trans = [0.8,0.8,0.8,0.0]
#     red = [1.0,0.0,0.0,1.0]
#     orange = [0.8, 0.258385, 0.041926, 1.0]
#     yellow = [0.4,0.4,0.1,1]
#     green = [0,1,0.2,0.7]
#     blue = [0.2,0.1,1,0.2]
#     default = orange
class OdentColors():
    white = [0.8,0.8,0.8,1.0]
    black = [0.0,0.0,0.0,1.0]
    trans = [0.8,0.8,0.8,0.0]
    red = [1.0,0.0,0.0,1.0]
    orange = [0.8, 0.258385, 0.041926, 1.0]
    yellow = [0.4,0.4,0.1,1]
    green = [0,1,0.2,0.7]
    blue = [0.0,0.5,0.8,0.1]
    olive = [0.8,0.6,0.0,0.7]
    base = [0.7,0.7,0.7,0.05]#[0.7,0.65,0.55,0.05]
    default = olive

def add_odent_libray() :
    message = []
    success = 0
    if not exists(OdentConstants.ODENT_LIBRARY_ARCHIVE_PATH):
        message = ["Odent Library archive not found"]
        return message,success
    _libs_collection_group = bpy.context.preferences.filepaths.asset_libraries
    odent_lib = _libs_collection_group.get(OdentConstants.ODENT_LIB_NAME)
    if odent_lib :
        idx = [i for i,l in enumerate(_libs_collection_group) if l.name==OdentConstants.ODENT_LIB_NAME][0]
        bpy.ops.preferences.asset_library_remove(index=idx)
    
    if exists(OdentConstants.ODENT_LIBRARY_PATH):
        shutil.rmtree(OdentConstants.ODENT_LIBRARY_PATH)
    os.mkdir(OdentConstants.ODENT_LIBRARY_PATH)

    archive_list = os.listdir(OdentConstants.ODENT_LIBRARY_ARCHIVE_PATH)
    for _item in archive_list :
        _item_full_path = join(OdentConstants.ODENT_LIBRARY_ARCHIVE_PATH,_item)
        if _item.endswith('.zip') :
            with zipfile.ZipFile(_item_full_path, 'r') as zip_ref:
                zip_ref.extractall(OdentConstants.ODENT_LIBRARY_PATH)
        else :
            shutil.copy2(_item_full_path, OdentConstants.ODENT_LIBRARY_PATH)

    bpy.ops.preferences.asset_library_add(directory=OdentConstants.ODENT_LIBRARY_PATH)
    success = 1
    return message, success

def import_required_modules(required_modules_dict=OdentConstants.REQ_DICT):
    missing_modules = []
    for mod, pkg in required_modules_dict.items():
        try:
            import_module(mod)
        except ImportError:
            missing_modules.append(pkg)

    return missing_modules

def isConnected(test_url="www.google.com",debug=False):
    result = False
    try:
        sock = socket.create_connection((test_url, 80))
        if sock is not None:
            sock.close
            result = True
        
    except OSError:
        pass

    if debug :
        info = "no connexion!"
        if result :
            info = "connected..."
        odent_log([info])
    return result

def browse(url) :
    success = 0
    try :
        webbrowser.open(url)
        success = 1
        return success
    except Exception as er :
        print(f"open telegram link error :\n{er}")
        return success

def start_blender_session():
    # print(f"binary path : {bpy.app.binary_path}")
    os.system(f'"{bpy.app.binary_path}"')



def set_modules_path(modules_path=OdentConstants.ODENT_MODULES_PATH):
    if not modules_path in sys.path :
        sys.path.insert(0,OdentConstants.ODENT_MODULES_PATH)

    
def addon_update_preinstall(update_root):
    
    update_data_map_json = join(update_root, OdentConstants.UPDATE_MAP_JSON)
    update_data_map_dict = open_json(update_data_map_json)
    update_data_dir = join(update_root, "data")
    items = os.listdir(update_data_dir)
    update_data_dict = {}
    for i in items:
        if update_data_map_dict.get(i):
            update_data_dict.update({join(update_data_dir,i): join(OdentConstants.ADDON_DIR,*update_data_map_dict.get(i))})
        else :
            odent_log([f"Update data {i} not found in update map!"])
            continue
    
    # update_data_dict = {join(update_data_dir,i) : join(OdentConstants.ADDON_DIR,*update_data_map_dict.get(i)) for i in items}
    for src,dst in update_data_dict.items():
        
        if OdentConstants.ODENT_MODULES_NAME in src.lower() :
            shutil.move(src, OdentConstants.RESOURCES)
        else :
            if not exists(dirname(dst)) :
                os.makedirs(dirname(dst))

            if exists(dst) :
                os.remove(dst) if isfile(dst) else shutil.rmtree(dst)
            
            shutil.move(src, dirname(dst))
        

def addon_update_download():
    
    message = []
    update_root = None 
    try :
        temp_dir = tempfile.mkdtemp()
        os.chdir(temp_dir)
        _update_zip_local = join(temp_dir,f'{OdentConstants.ADDON_UPDATE_NAME}.zip')

        # Download the file
        with requests.get(OdentConstants.ADDON_UPDATE_URL, stream=True, timeout=10) as r:
            try:
                r.raise_for_status()
            except HTTPError as http_err:
                txt = "HTTP error occurred"
                odent_log([f"{txt} : {http_err}"])
                message.extend(["Server connection error!"])
                return message,update_root
            except ConnectionError as conn_err:
                txt = "Server connection error!"
                odent_log([f"{txt} : {conn_err}"])
                message.extend([txt])
                return message,update_root
            except Timeout as timeout_err:
                txt = "Timeout error occurred"
                odent_log([f"{txt} : {timeout_err}"])
                message.extend(["Server connection error!"])
                return message,update_root
            except RequestException as req_err:
                txt = f"Error during requests to {OdentConstants.ADDON_UPDATE_URL}"
                odent_log([f"{txt} : {req_err}"])
                message.extend(["Server connection error!"])
                return message,update_root
            
            with open(_update_zip_local, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    except Exception as err:
        odent_log([f"An unexpected error occurred: {err}"]) 
        message.extend(["Server connection error!"])
        return message,update_root
        
        
    try :
        with zipfile.ZipFile(_update_zip_local, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        src = [abspath(e) for e in os.listdir(temp_dir) if isdir(abspath(e))][0]
        update_root = join(temp_dir,OdentConstants.ADDON_UPDATE_NAME)
        os.rename(src,update_root)
        return message,update_root
    
    except zipfile.BadZipFile as zip_err:
        txt = "Error occurred while extracting the downloaded addon ZIP file"
        odent_log([f"{txt} : {zip_err}"])
        message.extend([txt])
        return message,update_root
        
    except Exception as err:
        odent_log([f"Error occurred while extracting the downloaded addon ZIP file: {err}"]) 
        message.extend(["Update error!"])
        return message,update_root

def write_json(Dict,outPath) :
    jsonString = json.dumps(Dict,indent=4)
    with open(outPath, 'w') as wf :
        wf.write(jsonString)

def open_json(jsonPath) :
    with open(jsonPath, "r") as f:
        dataDict = json.load(f)
    return dataDict

def set_enum_items(items_list):
    return [(item, item, str(item)) for item in items_list]

def HuTo255(Hu, Wmin=OdentConstants.WMIN, Wmax= OdentConstants.WMAX):
    V255 = int(((Hu - Wmin) / (Wmax - Wmin)) * 255)
    return V255

def remove_handlers_by_names(handlers_names_list) :
    depsgraph_update_post_handlers = bpy.app.handlers.depsgraph_update_post
    frame_change_post_handlers = bpy.app.handlers.frame_change_post

    # Remove old handlers :
    depsgraph_h = [
        h for h in bpy.app.handlers.depsgraph_update_post if h.__name__ in handlers_names_list
    ]
    frame_change_h = [
        h for h in bpy.app.handlers.frame_change_post if h.__name__ in handlers_names_list
    ]
    
    for h in depsgraph_h:
        bpy.app.handlers.depsgraph_update_post.remove(h)
    for h in frame_change_h:
        bpy.app.handlers.frame_change_post.remove(h)
        
def add_handlers_from_func_list(func_list) :
    for h in func_list:
        bpy.app.handlers.depsgraph_update_post.append(h)
        bpy.app.handlers.frame_change_post.append(h)


class ODENT_GpuDrawText() :
    """gpu draw text in active area 3d"""
    
    global DRAW_HANDLERS

    def __init__(self,
                message_list = [],
                remove_handlers=True,
                button=False,
                percentage=100,
                redraw_timer=True,
                rect_color=OdentColors.default,
                txt_color = OdentColors.black,
                txt_size = OdentConstants.BLF_INFO.get("size"),
                btn_txt = "OK",
                info_handler = None,
                sleep_time=0
                ):
        
        
        self.message_list=message_list
        self.remove_handlers=remove_handlers
        self.button=button
        self.percentage=percentage
        self.redraw_timer=redraw_timer
        self.rect_color=rect_color
        
        self.txt_color=txt_color
        self.txt_size=txt_size
        self.btn_txt=btn_txt
        self.info_handler=info_handler
        self.rect_height=15
        self.line_height=2
        self.offset_vertical=35
        self.offset_horizontal=50
        self.sleep_time=sleep_time

        if self.remove_handlers:
            self._cancell_previous()
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP',iterations=1) 

        if self.message_list:
            self._cancell_previous()
            self.gpu_info_footer()
            DRAW_HANDLERS.append(self.info_handler)
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP',iterations=1)  

            if self.sleep_time != 0:
                sleep(self.sleep_time)
                self._cancell_previous()
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP',iterations=1) 
        else :
            if self.remove_handlers:
                self._cancell_previous()
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP',iterations=1) 
            

        # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP',iterations=1) 
    def _cancell_previous(self):
        for _h in DRAW_HANDLERS:
            try :
                bpy.types.SpaceView3D.draw_handler_remove(_h, "WINDOW")
                DRAW_HANDLERS.remove(_h)
            except Exception as er :
                odent_log([f"remove_handler error : {er}"])
         

    def gpu_info_footer(self):
        
        if self.percentage <= 0:
            self.percentage = 1
        if self.percentage > 100:
            self.percentage = 100

        def draw_callback_function():

            w = int(bpy.context.area.width * (self.percentage/100))
            w_full = bpy.context.area.width
            
            self.draw_gpu_rect(self.offset_horizontal, self.offset_vertical, w-self.offset_horizontal, self.line_height, self.rect_color)
            
            for i, txt in enumerate((reversed(self.message_list))):
                self.draw_gpu_rect(self.offset_horizontal, self.offset_vertical + self.line_height+2+(self.rect_height*i), w_full-self.offset_horizontal, self.rect_height, OdentColors.base)
                blf.position(OdentConstants.BLF_INFO.get('fontid'), self.offset_horizontal+2, self.offset_vertical + self.line_height+2+2+(self.rect_height*i), 0)
                blf.size(OdentConstants.BLF_INFO.get("fontid"), self.txt_size) # 3.6 api blf.size(0, 40, 30) -> blf.size(fontid, size)
                r, g, b, a = self.txt_color
                blf.color(0, r, g, b, a)
                blf.draw(0, txt)

            if self.button:
                self.draw_gpu_rect(w-110, 2, 100, self.rect_height-4, OdentColors.yellow)
                blf.position(0, w-85, 10, 0)
                blf.size(OdentConstants.BLF_INFO.get("fontid"), self.txt_size) # 3.6 api blf.size(0, 40, 30) -> blf.size(fontid, size)
                r, g, b, a = self.txt_color
                blf.color(0, r, g, b, a)
                blf.draw(0, self.btn_txt)

        self.info_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_function, (), "WINDOW", "POST_PIXEL"
        )
        
        # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        

    def draw_gpu_rect(self,x, y, w, h, rect_color):

        vertices = (
            (x, y), (x, y + h),
            (x + w, y + h), (x + w, y))

        indices = (
            (0, 1, 2), (0, 2, 3)
        )

        gpu.state.blend_set('ALPHA')
        shader = gpu.shader.from_builtin('UNIFORM_COLOR') # 3.6 api '2D_UNIFORM_COLOR'
        batch = batch_for_shader(
            shader, 'TRIS', {"pos": vertices}, indices=indices)
        shader.bind()
        shader.uniform_float("color", rect_color)
        batch.draw(shader)

class ODENT_OT_RemoveInfoFooter(bpy.types.Operator):
    """Remove Info Footer"""
    bl_idname = "wm.odent_remove_info_footer"
    bl_label = "Hide Info Footer"
    global DRAW_HANDLERS
    @classmethod
    def poll(cls, context):
        return DRAW_HANDLERS != []

    def execute(self, context):
        ODENT_GpuDrawText()
        return {"FINISHED"}
