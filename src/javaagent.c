#include "dbg.h"
#include "os.h"
#include "state.h"
#include <jni.h>
#include <jvmti.h>
#include "javabci.h"

static jmethodID g_mid_Object_hashCode          = NULL;
static jmethodID g_mid_SSLSocketImpl_getSession = NULL;
static jmethodID g_mid_AppOutputStream___write  = NULL;
static jfieldID  g_fid_AppOutputStream_socket   = NULL;
static jmethodID g_mid_AppInputStream___read    = NULL;
static jfieldID  g_fid_AppInputStream_socket    = NULL;

static void logJvmtiError(jvmtiEnv *jvmti, jvmtiError errnum, const char *str) {
    char buf[1024];
    char *errnum_str = NULL;
    (*jvmti)->GetErrorName(jvmti, errnum, &errnum_str);
    snprintf(buf, sizeof(buf), "ERROR: JVMTI: [%d] %s - %s\n", errnum, (errnum_str == NULL ? "Unknown": errnum_str), (str == NULL? "" : str));
    scopeLog(buf, -1, CFG_LOG_ERROR);
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
    if (name != NULL && strcmp(name, "sun/security/ssl/AppOutputStream") == 0) {
        scopeLog("installing Java SSL hooks for AppOutputStream class...", -1, CFG_LOG_DEBUG);

        jclass objectClass = (*jni)->FindClass(jni, "java/lang/Object");
        g_mid_Object_hashCode = (*jni)->GetMethodID(jni, objectClass, "hashCode", "()I");

        jclass sslSocketImplClass = (*jni)->FindClass(jni, "sun/security/ssl/SSLSocketImpl");
        g_mid_SSLSocketImpl_getSession = (*jni)->GetMethodID(jni, sslSocketImplClass, "getSession", "()Ljavax/net/ssl/SSLSession;");
    
        java_class_t *classInfo = javaReadClass(class_data);

        int methodIndex = javaFindMethodIndex(classInfo, "write", "([BII)V");
        javaCopyMethod(classInfo, classInfo->methods[methodIndex], "__write");
        javaConvertMethodToNative(classInfo, methodIndex);

        unsigned char *dest;
        (*jvmti_env)->Allocate(jvmti_env, classInfo->length, &dest);
        javaWriteClass(dest, classInfo);

        *new_class_data_len = classInfo->length;
        *new_class_data = dest;
        javaDestroy(&classInfo);
    }

    if (name != NULL && strcmp(name, "sun/security/ssl/AppInputStream") == 0) {
        scopeLog("installing Java SSL hooks for AppInputStream class...", -1, CFG_LOG_DEBUG);

        java_class_t *classInfo = javaReadClass(class_data);

        int methodIndex = javaFindMethodIndex(classInfo, "read", "([BII)I");
        javaCopyMethod(classInfo, classInfo->methods[methodIndex], "__read");
        javaConvertMethodToNative(classInfo, methodIndex);

        unsigned char *dest;
        (*jvmti_env)->Allocate(jvmti_env, classInfo->length, &dest);
        javaWriteClass(dest, classInfo);

        *new_class_data_len = classInfo->length;
        *new_class_data = dest;
        javaDestroy(&classInfo);
    }
}

static void 
doJavaProtocol(JNIEnv *jni, jobject socket, jbyteArray buf, jint offset, jint len, metric_t src)
{
    jobject session   = (*jni)->CallObjectMethod(jni, socket, g_mid_SSLSocketImpl_getSession);
    jint    hash      = (*jni)->CallIntMethod(jni, session, g_mid_Object_hashCode);
    jbyte   *byteBuf  = (*jni)->GetPrimitiveArrayCritical(jni, buf, 0);
    doProtocol((uint64_t)hash, -1, &byteBuf[offset], (size_t)(len - offset), src, BUF);
    (*jni)->ReleasePrimitiveArrayCritical(jni, buf, buf, 0);
}

JNIEXPORT void JNICALL 
Java_sun_security_ssl_AppOutputStream_write(JNIEnv *jni, jobject obj, jbyteArray buf, jint offset, jint len) 
{
    if (g_mid_AppOutputStream___write == NULL) {
        jclass appOutputStreamClass   = (*jni)->FindClass(jni, "sun/security/ssl/AppOutputStream");
        g_mid_AppOutputStream___write = (*jni)->GetMethodID(jni, appOutputStreamClass, "__write", "([BII)V");
        g_fid_AppOutputStream_socket  = (*jni)->GetFieldID(jni, appOutputStreamClass, "c", "Lsun/security/ssl/SSLSocketImpl;");
    }

    jobject socket  = (*jni)->GetObjectField(jni, obj, g_fid_AppOutputStream_socket);
    doJavaProtocol(jni, socket, buf, offset, len, TLSTX);
    
    //call the original method
    (*jni)->CallVoidMethod(jni, obj, g_mid_AppOutputStream___write, buf, offset, len);
} 

JNIEXPORT jint JNICALL 
Java_sun_security_ssl_AppInputStream_read(JNIEnv *jni, jobject obj, jbyteArray buf, jint offset, jint len) 
{
    if (g_mid_AppInputStream___read == NULL) {
        jclass appInputStreamClass   = (*jni)->FindClass(jni, "sun/security/ssl/AppInputStream");
        g_mid_AppInputStream___read  = (*jni)->GetMethodID(jni, appInputStreamClass, "__read", "([BII)I");
        g_fid_AppInputStream_socket  = (*jni)->GetFieldID(jni, appInputStreamClass, "c", "Lsun/security/ssl/SSLSocketImpl;");
    }

    //call the original method
    jint res = (*jni)->CallIntMethod(jni, obj, g_mid_AppInputStream___read, buf, offset, len);

    jobject socket = (*jni)->GetObjectField(jni, obj, g_fid_AppInputStream_socket);
    doJavaProtocol(jni, socket, buf, offset, res, TLSRX);

    return res;
}

JNIEXPORT jint JNICALL 
Agent_OnLoad(JavaVM *jvm, char *options, void *reserved) 
{
    jvmtiError error;
    jvmtiEnv *env;

    jint result = (*jvm)->GetEnv(jvm, (void **) &env, JVMTI_VERSION_1_0);
    if (result != 0) {
        scopeLog("ERROR: GetEnv failed\n", -1, CFG_LOG_ERROR);
        return JNI_ERR;
    }

    jvmtiCapabilities capabilities;
    memset(&capabilities,0, sizeof(capabilities));

    capabilities.can_generate_all_class_hook_events = 1;
    error = (*env)->AddCapabilities(env, &capabilities);
    if (error != JVMTI_ERROR_NONE) {
        logJvmtiError(env, error, "AddCapabilities");
        return JNI_ERR;
    }

    error = (*env)->SetEventNotificationMode(env, JVMTI_ENABLE, JVMTI_EVENT_CLASS_FILE_LOAD_HOOK, NULL);
    if (error != JVMTI_ERROR_NONE) {
        logJvmtiError(env, error, "SetEventNotificationMode");
        return JNI_ERR;
    }
   
    jvmtiEventCallbacks callbacks;
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.ClassFileLoadHook = &ClassFileLoadHook;
    error = (*env)->SetEventCallbacks(env, &callbacks, sizeof(callbacks));
    if (error != JVMTI_ERROR_NONE) {
        logJvmtiError(env, error, "SetEventCallbacks");
        return JNI_ERR;
    }
    
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