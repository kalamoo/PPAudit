import json
import time
from androguard.misc import AnalyzeAPK
from androguard.core.androconf import load_api_specific_resource_module
#  pip install androguard[magic,GUI]==3.3.5
from pathlib import Path


class evidenceExtractor:
    def __init__(self):
        self.permission_data_map = {
            "android.permission.ACCESS_BACKGROUND_LOCATION": [
                "geo location"
            ],
            "android.permission.ACCESS_COARSE_LOCATION": [
                "geo location"
            ],
            "android.permission.ACCESS_FINE_LOCATION": [
                "geo location"
            ],
            "android.permission.ACCESS_MEDIA_LOCATION": [
                "geo location"
            ],
            "android.permission.ACCESS_NETWORK_STATE": [
                "network"
            ],
            "android.permission.ACCESS_NOTIFICATION_POLICY": [],
            "android.permission.ACCESS_WIFI_STATE": [
                "network"
            ],
            "android.permission.AUTHENTICATE_ACCOUNTS": [
                "account"
            ],
            "android.permission.BLUETOOTH": [
                "network"
            ],
            "android.permission.BLUETOOTH_ADVERTISE": [
                "network"
            ],
            "android.permission.BLUETOOTH_ADMIN": [
                "network"
            ],
            "android.permission.BLUETOOTH_CONNECT": [
                "network"
            ],
            "android.permission.BLUETOOTH_SCAN": [
                "network"
            ],
            "android.permission.BROADCAST_STICKY": [],
            "android.permission.CAMERA": [
                "camera"
            ],
            "android.permission.camera2.full": [
                "camera"
            ],
            "android.permission.CHANGE_NETWORK_STATE": [
                "network"
            ],
            "android.permission.CHANGE_WIFI_MULTICAST_STATE": [
                "network"
            ],
            "android.permission.CHANGE_WIFI_STATE": [
                "network"
            ],
            "android.permission.CONTROL_LOCATION_UPDATES": [
                "geo location"
            ],
            "android.permission.CREDENTIAL_MANAGER_QUERY_CANDIDATE_CREDENTIALS": [],
            "android.permission.CREDENTIAL_MANAGER_SET_ALLOWED_PROVIDERS": [],
            "android.permission.CREDENTIAL_MANAGER_SET_ORIGIN": [],
            "android.permission.DEVICE_POWER": [
                "battery"
            ],
            "android.permission.DOWNLOAD_WITHOUT_NOTIFICATION": [],
            "android.permission.FLASHLIGHT": [],
            "android.permission.FOREGROUND_SERVICE_MICROPHONE": [
                "audio"
            ],
            "android.permission.FOREGROUND_SERVICE": [
                "usage info"
            ],
            "android.permission.FOREGROUND_SERVICE_CAMERA": [
                "camera"
            ],
            "android.permission.FOREGROUND_SERVICE_DATA_SYNC": [],
            "android.permission.FOREGROUND_SERVICE_MEDIA_PLAYBACK": [],
            "android.permission.GET_ACCOUNTS": [
                "account"
            ],
            "android.permission.GET_TASKS": [
                "usage info"
            ],
            "android.permission.INJECT_EVENTS": [],
            "android.permission.INSTALL_PACKAGES": [
                "usage info"
            ],
            "android.permission.INTERNET": [
                "network"
            ],
            "android.permission.INTERACT_ACROSS_USERS": [],
            "android.permission.INTERACT_ACROSS_USERS_FULL": [],
            "android.permission.KILL_BACKGROUND_PROCESSES": [],
            "android.permission.NEARBY_WIFI_DEVICES": [
                "network"
            ],
            "android.permission.READ_PHONE_NUMBERS": [
                # "phone num",
                "device info"
            ],
            "android.permission.READ_SYNC_STATS": [],
            "android.permission.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS": [
                "battery"
            ],
            "com.android.launcher.permission.INSTALL_SHORTCUT": [],
            "com.android.launcher.permission.UNINSTALL_SHORTCUT": [],
            "com.chrome.permission.DEVICE_EXTRAS": [],
            "android.permission.MANAGE_ACCOUNTS": [
                "account"
            ],
            "android.permission.MANAGE_EXTERNAL_STORAGE": [
                "information"
            ],
            "android.permission.MANAGE_OWN_CALLS": [],
            "android.permission.MICROPHONE": [
                "audio"
            ],
            "android.permission.MODIFY_AUDIO_SETTINGS": [],
            "android.permission.MOUNT_UNMOUNT_FILESYSTEMS": [],
            "android.permission.NFC": [],
            "android.permission.INSTALL_SHORTCUT": [],
            "android.permission.PACKAGE_USAGE_STATS": [
                "usage info"
            ],
            "android.permission.POST_NOTIFICATIONS": [],
            "android.permission.QUERY_ALL_PACKAGES": [
                "usage info"
            ],
            "android.permission.READ_CONTACTS": [
                "contact"
            ],
            "android.permission.READ_EXTERNAL_STORAGE": [
                "information"
            ],
            "android.permission.READ_INTERNAL_STORAGE": [],
            "android.permission.READ_LOGS": [
                "information"
            ],
            "android.permission.READ_MEDIA_AUDIO": [
                "audio"
            ],
            "android.permission.READ_MEDIA_IMAGES": [
                "camera"
            ],
            "android.permission.READ_MEDIA_VIDEO": [
                "camera"
            ],
            "android.permission.READ_MEDIA_VISUAL_USER_SELECTED": [
                "camera",
                "audio"
            ],
            "android.permission.READ_PHONE_STATE": [
                # "phone num",
                "device info"
            ],
            "android.permission.READ_PROFILE": [
                "account"
            ],
            "android.permission.READ_SYNC_SETTINGS": [],
            "android.permission.READ_WRITE_STORAGE": [
                "information"
            ],
            "android.permission.RECEIVE_BOOT_COMPLETED": [],
            "android.permission.RECORD_AUDIO": [
                "audio"
            ],
            "android.permission.REQUEST_COMPANION_USE_DATA_IN_BACKGROUND": [],
            "android.permission.REQUEST_COMPANION_RUN_IN_BACKGROUND": [],
            "android.permission.REQUEST_DELETE_PACKAGES": [
                "usage info"
            ],
            "android.permission.REQUEST_INSTALL_PACKAGES": [
                'usage info'
            ],
            "android.permission.REORDER_TASKS": [
                "usage info"
            ],
            "android.permission.REQUEST_VIDEO_AUDIO_CODE": [
                "audio",
                "camera",
            ],
            "android.permission.RESTART_PACKAGES": [],
            "android.permission.SCHEDULE_EXACT_ALARM": [],
            "android.permission.SEND_SMS": [
                "network"
            ],
            "android.permission.RECEIVE_SMS": [
                "network"
            ],
            "android.permission.SET_TIME_ZONE": [
                "geo location"
            ],
            "android.permission.SYSTEM_ALERT_WINDOW": [],
            "android.permission.USE_BIOMETRIC": [
                "biometric"
            ],
            "android.permission.USE_CREDENTIALS": [
                "account"
            ],
            "android.permission.USE_FINGERPRINT": [
                "fingerprint"
            ],
            "android.permission.USE_FULL_SCREEN_INTENT": [],
            "android.permission.VIBRATE": [
                "vibrator"
            ],
            "android.permission.VOICE_RECORD": [
                "audio"
            ],
            "android.permission.WAKE_LOCK": [],
            "android.permission.WRITE_INTERNAL_STORAGE": [],
            "android.permission.Write_EXTERNAL_STORAGE": [
                "information"
            ],
            "Android.permission.WRITE_EXTERNAL_STORAGE": [
                "information"
            ],
            "android.permission.WRITE_EXTERNAL_STORAGE": [
                "information"
            ],
            "android.permission.WRITE_CONTACTS": [
                "contact"
            ],
            "android.permission.WRITE_SYNC_SETTINGS": [],
            "android.permission.WRITE_SECURE_SETTINGS": [],
            "android.permission.WRITE_SETTINGS": [],
            "android.permission.USB_PERMISSION": [],
            "com.amazon.device.permission.COMRADE_CAPABILITIES": [],
            "com.amazon.permission.media.session.voicecommandcontrol": [
                "audio"
            ],
            "com.amazonaws.unity.permission.C2D_MESSAGE": [],
            "com.android.permission.READ_EXTERNAL_STORAGE": [],
            "com.android.permission.WRITE_EXTERNAL_STORAGE": [
                "information"
            ],
            "com.android.providers.tv.permission.READ_EPG_DATA": [],
            "com.android.providers.tv.permission.WRITE_EPG_DATA": [],
            "com.android.vending.BILLING": [
                "billing"
            ],
            "com.android.vending.CHECK_LICENSE": [
                "billing"
            ],
            "com.echoboom.dogfightelite.permission.C2D_MESSAGE": [],
            "com.facebook.services.identity.FEO2": [
                "account"
            ],
            "com.google.android.apps.now.CURRENT_ACCOUNT_ACCESS": [
                "account"
            ],
            "com.google.android.c2dm.permission.RECEIVE": [],
            "com.google.android.finsky.permission.BIND_GET_INSTALL_REFERRER_SERVICE": [],
            "com.google.android.gms.permission.AD_ID": [
                "ad id"
            ],
            "com.google.android.providers.gsf.permission.READ_GSERVICES": [
                "geo location"
            ],
            "com.holodia.holofit.permission.C2D_MESSAGE": [],
            "com.htc.launcher.permission.READ_SETTINGS": [],
            "com.htc.launcher.permission.UPDATE_SHORTCUT": [],
            "com.huawei.android.launcher.permission.CHANGE_BADGE": [],
            "com.huawei.android.launcher.permission.READ_SETTINGS": [],
            "com.huawei.android.launcher.permission.WRITE_SETTINGS": [],
            "com.infiniteMonkey.BlocksSimulator.WRITE_EXTERNAL_STORAGE": [
                "information"
            ],
            "com.igalia.wolvic.metastore.CRASH_RECEIVER_PERMISSION": [
                "error report",
            ],
            "com.oculus.ovrmonitormetricsservice.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION": [],
            "uk.co.digitalnauts.HoloHubVRClient.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION": [],
            "com.oculus.permission.ACCESS_TRACKING_ENV": [
                "environment",
                "camera",
                "vr play area"
            ],
            "com.oculus.permission.BODY_TRACKING": [
                "body measure",
                "camera",
                "gyroscope",
                "accelerometer",
                "vr movement",
                "infrared",
                "vr play area",
                "environment"
            ],
            "com.oculus.permission.EYE_TRACKING": [
                "eye tracking",
                "pupil distance",
                "camera",
                "infrared"
            ],
            "com.oculus.permission.FACE_TRACKING": [
                "face",
                "camera",
                "infrared"
            ],
            "com.oculus.permission.HAND_TRACKING": [
                "hand tracking",
                "camera"
            ],
            "com.oculus.permission.IMPORT_EXPORT_IOT_MAP_DATA": [
                "environment",
                "camera",
                "vr play area"
            ],
            "com.oculus.permission.PLAY_AUDIO_BACKGROUND": [],
            "com.oculus.permission.RECORD_AUDIO_BACKGROUND": [
                "audio"
            ],
            "com.oculus.permission.RENDER_MODEL": [],
            "com.oculus.permission.REPORT_EVENTS": [
                "error report"
            ],
            "com.oculus.permission.REPORT_EVENTS_DEBUG": [
                "error report"
            ],
            "com.oculus.permission.TRACKED_KEYBOARD": [
                "camera"
            ],
            "com.oculus.permission.USE_ANCHOR_API": [
                "camera",
                "vr play area"
            ],
            "com.oculus.permission.WIFI_LOCK": [
                "network"
            ],
            "com.samsung.android.hmt.permission.READ_SETTINGS": [],
            "com.samsung.permission.HRM_EXT": [
                # https://developer.samsung.com/galaxy-sensor-extension/overview.html;
                # https://stackoverflow.com/questions/49628118/app-crashes-with-sqliteconstraintexception-but-i-am-not-using-sqlite
                "heart rate"
            ],
            "com.samsung.permission.SSENSOR": [
                # https://developer.samsung.com/galaxy-sensor-extension/overview.html;
                # https://stackoverflow.com/questions/49628118/app-crashes-with-sqliteconstraintexception-but-i-am-not-using-sqlite
                "heart rate"
            ],
            "com.sec.android.provider.badge.permission.READ": [],
            "com.sec.android.provider.badge.permission.WRITE": [],
            "com.sonyericsson.home.permission.BROADCAST_BADGE": [],
            "com.sonymobile.home.permission.PROVIDER_INSERT_BADGE": [],
            "com.whatsapp.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION": [],
            "com.whatsapp.permission.BROADCAST": [],
            "com.whatsapp.permission.MAPS_RECEIVE": [
                "geo location"
            ],
            "com.whatsapp.permission.REGISTRATION": [],
            "com.whatsapp.sticker.READ": [],
            "com.xiaomi.permission.AUTH_SERVICE": [
                "account"
            ],
            "com.xiaomi.sdk.permission.PAYMENT": [
                "billing"
            ],
            "com.tdv.vrapp.CRASH_RECEIVER_PERMISSION": [
                "error report"
            ],
            "com.xr.AccessIrisCoreService": [
                # https://open.oppomobile.com/ar/help/0.7.1.html
                "eye tracking"
            ],
            "com.xrspace.xraccount.permission.SERVICE_ACCESS": [
                "device info"
            ],
            "jp.actevolve.vrpf_liveedit.permission.C2D_MESSAGE": [],
            "oculus.permission.HAND_TRACKING": [
                "hand tracking",
                "camera"
            ],
            "oculus.permission.handtracking": [
                "hand tracking",
                "camera"
            ],
            "oculus.render_notification": [],
            "org.chromium.chrome.permission.C2D_MESSAGE": [],
            "org.chromium.chrome.permission.READ_WRITE_BOOKMARK_FOLDERS": [
                "usage info"
            ],
            "org.chromium.chrome.TOS_ACKED": [],
            "org.mozilla.vrbrowser.CRASH_RECEIVER_PERMISSION": [
                "error report"
            ],
            "questpatcher.modded": [],
            "vive.wave.vr.oem.data.OEMDataWrite": [],
            "vive.wave.vr.oem.data.OEMDataRead": []
        }

        self.api_permission_map = load_api_specific_resource_module("api_permission_mappings")
    
    def permissions_to_data(self, permissions):
        data_objs = []
        for permission in permissions:
            data_objs.extend(self.permission_data_map[permission])
        return set(data_objs)
    
    def api_to_permissions(self, dx):
        results = []
        for method_analysis in dx.get_methods():
            method = method_analysis.get_method()
            name = method.get_class_name() + "-" \
                + method.get_name() + "-" \
                + str(method.get_descriptor())
            # Name example:
            # Landroid/media/AudioManager;-
            # setSpeakerphoneOn-
            # (Z)V
            for k, v in self.api_permission_map.items():
                if name == k:
                    # result = str(method) + ": " + str(v)
                    if (k, v) not in results:
                        results.append((k, v))
        return results
    
    def mapping_to_data(self, evidence_dict):
        permissions = evidence_dict['from_xml'] + evidence_dict['from_api']
        return list(self.permissions_to_data(permissions))

    def extract_evidence_single(self, apk_file):
        try:
            a, d, dx = AnalyzeAPK(apk_file)
            permissions_from_xml = a.get_permissions()
            apis_and_permissions_from_api = self.api_to_permissions(dx)
            evidences = {
                'from_xml': permissions_from_xml,
                'from_api': apis_and_permissions_from_api
            }
        except Exception as e:
            print('[Error {}] {}'.format(apk_file, str(e)))
        return evidences