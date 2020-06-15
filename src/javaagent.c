#include "dbg.h"
#include "os.h"
#include "state.h"
#include <jni.h>
#include <jvmti.h>
#include "javabci.h"

static void check_error(jvmtiEnv *jvmti, jvmtiError errnum, const char *str) {
    if (errnum != JVMTI_ERROR_NONE) {
        char *errnum_str = NULL;
        (void) (*jvmti)->GetErrorName(jvmti, errnum, &errnum_str);
        printf("ERROR: JVMTI: [%d] %s - %s\n", errnum, (errnum_str == NULL ? "Unknown": errnum_str), (str == NULL? "" : str));
    }
}

void JNICALL 
ClassFileLoadHook(jvmtiEnv *jvmti_env,
    JNIEnv* jni_env,
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

static void printBuf(JNIEnv* jni, jobject buf_param, char *method) {
  jint len = (*jni)->GetArrayLength(jni, buf_param);
  printf("METHOD = %s, LEN = %d\nBUFFER=", method, len);

  jbyte *buf = (*jni)->GetPrimitiveArrayCritical(jni, buf_param, 0);
  for(int i=0; i<len; i++) {
    if (buf[i]>31 && buf[i]<127) printf("%c", buf[i]);
  }
  printf("\n\n");
  (*jni)->ReleasePrimitiveArrayCritical(jni, buf_param, buf, 0);
}

JNIEXPORT void JNICALL 
Java_sun_security_ssl_SSLSocketImpl_readDataRecord(JNIEnv *jni, jobject obj, jobject inputRecord) 
{
    jclass clazz  = (*jni)->FindClass(jni, "sun/security/ssl/SSLSocketImpl");
    jclass inputRecordClass  = (*jni)->FindClass(jni, "sun/security/ssl/InputRecord");

    jmethodID mid = (*jni)->GetMethodID(jni, clazz, "__readDataRecord", "(Lsun/security/ssl/InputRecord;)V");
    (*jni)->CallVoidMethod(jni, obj, mid, inputRecord);

    jfieldID fid = (*jni)->GetFieldID(jni, inputRecordClass, "buf", "[B");
    jbyteArray buf = (*jni)->GetObjectField(jni, inputRecord, fid);

    jfieldID fid2 = (*jni)->GetFieldID(jni, inputRecordClass, "pos", "I");
    jint offset = (*jni)->GetIntField(jni, inputRecord, fid2);

    jfieldID fid3 = (*jni)->GetFieldID(jni, inputRecordClass, "count", "I");
    jint count = (*jni)->GetIntField(jni, inputRecord, fid3);

    jclass objectClass  = (*jni)->FindClass(jni, "java/lang/Object");

    jmethodID hashCodeId = (*jni)->GetMethodID(jni, objectClass, "hashCode", "()I");
    jint hash = (*jni)->CallIntMethod(jni, obj, hashCodeId);

    
    jbyte *byteBuf = (*jni)->GetPrimitiveArrayCritical(jni, buf, 0);
    doProtocol((uint64_t)hash, -1, &byteBuf[offset], (size_t)(count - offset), TLSRX, BUF);
    (*jni)->ReleasePrimitiveArrayCritical(jni, buf, buf, 0);
}

JNIEXPORT void JNICALL 
Java_sun_security_ssl_SSLSocketImpl_writeRecord(JNIEnv *jni, jobject obj, jobject inputRecord, jboolean boolarg) 
{
    jclass clazz  = (*jni)->FindClass(jni, "sun/security/ssl/SSLSocketImpl");
    jclass inputRecordClass  = (*jni)->FindClass(jni, "sun/security/ssl/OutputRecord");

    jfieldID fid = (*jni)->GetFieldID(jni, inputRecordClass, "buf", "[B");
    jbyteArray buf = (*jni)->GetObjectField(jni, inputRecord, fid);

    jfieldID fid3 = (*jni)->GetFieldID(jni, inputRecordClass, "count", "I");
    jint count = (*jni)->GetIntField(jni, inputRecord, fid3);

    jclass objectClass  = (*jni)->FindClass(jni, "java/lang/Object");

    jmethodID hashCodeId = (*jni)->GetMethodID(jni, objectClass, "hashCode", "()I");
    jint hash = (*jni)->CallIntMethod(jni, obj, hashCodeId);

    jbyte *byteBuf = (*jni)->GetPrimitiveArrayCritical(jni, buf, 0);
    doProtocol((uint64_t)hash, -1, &byteBuf[0], (size_t)(count), TLSTX, BUF);
    (*jni)->ReleasePrimitiveArrayCritical(jni, buf, buf, 0);

    jmethodID mid = (*jni)->GetMethodID(jni, clazz, "__writeRecord", "(Lsun/security/ssl/OutputRecord;Z)V");
    (*jni)->CallVoidMethod(jni, obj, mid, inputRecord, boolarg);
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