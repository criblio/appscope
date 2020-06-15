#include "dbg.h"
#include "os.h"
#include "state.h"
#include <jni.h>
#include <jvmti.h>
#include "javabci.h"

static jmethodID g_mid_Object_hashCode = NULL;
static jfieldID g_fid_InputRecord_buf = NULL;
static jfieldID g_fid_InputRecord_pos = NULL;
static jfieldID g_fid_InputRecord_count = NULL;
static jmethodID g_mid_InputRecord_contentType = NULL;

static jfieldID g_fid_OutputRecord_buf = NULL;
static jfieldID g_fid_OutputRecord_count = NULL;
static jmethodID g_mid_OutputRecord_contentType = NULL;
static jint g_record_headerSize = 0;

#define CT_APPLICATION_DATA 23

static void check_error(jvmtiEnv *jvmti, jvmtiError errnum, const char *str) {
    if (errnum != JVMTI_ERROR_NONE) {
        char *errnum_str = NULL;
        (void) (*jvmti)->GetErrorName(jvmti, errnum, &errnum_str);
        printf("ERROR: JVMTI: [%d] %s - %s\n", errnum, (errnum_str == NULL ? "Unknown": errnum_str), (str == NULL? "" : str));
    }
}

void JNICALL 
ClassFileLoadHook(jvmtiEnv *jvmti_env,
    JNIEnv* jni,
    jclass class_being_redefined,
    jobject loader,
    const char* name,
    jobject protection_domain,
    jint class_data_len,
    const unsigned char* class_data,
    jint* new_class_data_len,
    unsigned char** new_class_data) 
{
    
    if (name != NULL && strcmp(name, "sun/security/ssl/SSLSocketImpl") == 0) {
        scopeLog("installing Java SSL hooks...", -1, CFG_LOG_DEBUG);

        jclass objectClass  = (*jni)->FindClass(jni, "java/lang/Object");
        g_mid_Object_hashCode = (*jni)->GetMethodID(jni, objectClass, "hashCode", "()I");

        jclass recordClass  = (*jni)->FindClass(jni, "sun/security/ssl/Record");
        jfieldID fid = (*jni)->GetStaticFieldID(jni, recordClass, "headerSize", "I");
        g_record_headerSize = (*jni)->GetStaticIntField(jni, recordClass, fid);

        jclass inputRecordClass  = (*jni)->FindClass(jni, "sun/security/ssl/InputRecord");
        g_mid_InputRecord_contentType = (*jni)->GetMethodID(jni, inputRecordClass, "contentType", "()B");
        g_fid_InputRecord_buf = (*jni)->GetFieldID(jni, inputRecordClass, "buf", "[B");
        g_fid_InputRecord_pos = (*jni)->GetFieldID(jni, inputRecordClass, "pos", "I");
        g_fid_InputRecord_count = (*jni)->GetFieldID(jni, inputRecordClass, "count", "I");

        jclass outputRecordClass  = (*jni)->FindClass(jni, "sun/security/ssl/OutputRecord");
        g_mid_OutputRecord_contentType = (*jni)->GetMethodID(jni, outputRecordClass, "contentType", "()B");
        g_fid_OutputRecord_buf = (*jni)->GetFieldID(jni, outputRecordClass, "buf", "[B");
        g_fid_OutputRecord_count = (*jni)->GetFieldID(jni, outputRecordClass, "count", "I");
    
        //modify SSLSocketImpl class
        java_class_t *classInfo = javaReadClass((void *)class_data);

        int methodIndex = javaFindMethodIndex(classInfo, "readDataRecord", "(Lsun/security/ssl/InputRecord;)V");
        javaCopyMethod(classInfo, classInfo->methods[methodIndex], "__readDataRecord");
        javaConvertMethodToNative(classInfo, methodIndex);

        methodIndex = javaFindMethodIndex(classInfo, "writeRecord", "(Lsun/security/ssl/OutputRecord;Z)V");
        javaCopyMethod(classInfo, classInfo->methods[methodIndex], "__writeRecord");
        javaConvertMethodToNative(classInfo, methodIndex);

        unsigned char *dest;
        (*jvmti_env)->Allocate(jvmti_env, classInfo->length, &dest);
        javaWriteClass(dest, classInfo);

        *new_class_data_len = classInfo->length;
        *new_class_data = dest;
        javaDestroy(&classInfo);
    }
}

// static void printBuf(JNIEnv* jni, jobject buf_param, char *method, int offset, int len) {
//     //jint len = (*jni)->GetArrayLength(jni, buf_param);
//     printf("METHOD = %s, LEN = %d. OFFSET = %d \nBUFFER=", method, len, offset);

//     jbyte *buf = (*jni)->GetPrimitiveArrayCritical(jni, buf_param, 0);
//     for(int i=0; i<len; i++) {
//         printf("%c", buf[i + offset]);
//     }
//     printf("\n\n");
//     (*jni)->ReleasePrimitiveArrayCritical(jni, buf_param, buf, 0);
// }

JNIEXPORT void JNICALL 
Java_sun_security_ssl_SSLSocketImpl_readDataRecord(JNIEnv *jni, jobject obj, jobject inputRecord) 
{
    jclass clazz   = (*jni)->FindClass(jni, "sun/security/ssl/SSLSocketImpl");
    jmethodID mid  = (*jni)->GetMethodID(jni, clazz, "__readDataRecord", "(Lsun/security/ssl/InputRecord;)V");

    //call the original method
    (*jni)->CallVoidMethod(jni, obj, mid, inputRecord);

    jbyte contentType = (*jni)->CallByteMethod(jni, inputRecord, g_mid_InputRecord_contentType);
    if (contentType == CT_APPLICATION_DATA) {
        jbyteArray buf = (*jni)->GetObjectField(jni, inputRecord, g_fid_InputRecord_buf);
        jint offset    = (*jni)->GetIntField(jni, inputRecord, g_fid_InputRecord_pos);
        jint count     = (*jni)->GetIntField(jni, inputRecord, g_fid_InputRecord_count);
        jint hash      = (*jni)->CallIntMethod(jni, obj, g_mid_Object_hashCode);
        jbyte *byteBuf = (*jni)->GetPrimitiveArrayCritical(jni, buf, 0);
        doProtocol((uint64_t)hash, -1, &byteBuf[offset], (size_t)(count - offset), TLSRX, BUF);
        (*jni)->ReleasePrimitiveArrayCritical(jni, buf, buf, 0);
    }
}

JNIEXPORT void JNICALL 
Java_sun_security_ssl_SSLSocketImpl_writeRecord(JNIEnv *jni, jobject obj, jobject outputRecord, jboolean boolArg) 
{
    jbyte contentType = (*jni)->CallByteMethod(jni, outputRecord, g_mid_OutputRecord_contentType);
    if (contentType == CT_APPLICATION_DATA) {
        jbyteArray buf    = (*jni)->GetObjectField(jni, outputRecord, g_fid_OutputRecord_buf);
        jint count        = (*jni)->GetIntField(jni, outputRecord, g_fid_OutputRecord_count);
        jint hash         = (*jni)->CallIntMethod(jni, obj, g_mid_Object_hashCode);
        jbyte *byteBuf    = (*jni)->GetPrimitiveArrayCritical(jni, buf, 0);
        jint headerSize   = g_record_headerSize;

        //skip zeros
        while(byteBuf[headerSize]==0 && headerSize<count) headerSize++;

        doProtocol((uint64_t)hash, -1, &byteBuf[headerSize], (size_t)(count - headerSize), TLSTX, BUF);
        (*jni)->ReleasePrimitiveArrayCritical(jni, buf, buf, 0);
    }

    //call the original method
    jclass clazz  = (*jni)->FindClass(jni, "sun/security/ssl/SSLSocketImpl");
    jmethodID mid = (*jni)->GetMethodID(jni, clazz, "__writeRecord", "(Lsun/security/ssl/OutputRecord;Z)V");
    (*jni)->CallVoidMethod(jni, obj, mid, outputRecord, boolArg);
}

JNIEXPORT jint JNICALL 
Agent_OnLoad(JavaVM *jvm, char *options, void *reserved) 
{
    jvmtiError error;
    jint result;
    jvmtiEnv *env;
    jvmtiEventCallbacks callbacks;

    result = (*jvm)->GetEnv(jvm, (void **) &env, JVMTI_VERSION_1_0);
    if (result != 0) {
        scopeLog("ERROR: GetEnv failed\n", -1, CFG_LOG_ERROR);
        return JNI_ERR;
    }

    jvmtiCapabilities capabilities;
    memset(&capabilities,0, sizeof(capabilities));

    capabilities.can_generate_all_class_hook_events = 1;
    error = (*env)->AddCapabilities(env, &capabilities);
    check_error(env, error, "AddCapabilities");

    error = (*env)->SetEventNotificationMode(env, JVMTI_ENABLE, JVMTI_EVENT_CLASS_FILE_LOAD_HOOK, NULL);
    check_error(env, error, "SetEventNotificationMode->JVMTI_EVENT_CLASS_FILE_LOAD_HOOK");

    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.ClassFileLoadHook = &ClassFileLoadHook;
    error = (*env)->SetEventCallbacks(env, &callbacks, sizeof(callbacks));
    check_error(env, error, "SetEventCallbacks");
    
    return JNI_OK;
}

void
initJavaAgent() {
    //TODO: 
    // - check if we are in a java process
    // - preserve existing _JAVA_OPTIONS
    char *var = getenv("LD_PRELOAD");
    if (var != NULL) {
        char buf[1024];
        snprintf(buf, sizeof(buf), "-agentpath:%s", var);
        int result = setenv("_JAVA_OPTIONS", buf, 1);
        if (result) {
            scopeLog("ERROR: Could not set _JAVA_OPTIONS failed\n", -1, CFG_LOG_ERROR);
        }
    }
}