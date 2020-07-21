#include <torch/csrc/cupti_tracer.h>

#include <cstdio>
#include <vector>
#include <stdlib.h>
#include <memory>
#include <string.h>

#include <cupti.h>

#include <Python.h>

#define CUPTI_CALL(call)                                                         \
    do {                                                                         \
        CUptiResult _status = call;                                              \
        if (_status != CUPTI_SUCCESS) {                                          \
            const char *errstr;                                                  \
            cuptiGetResultString(_status, &errstr);                              \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                    __FILE__, __LINE__, #call, errstr);                          \
            exit(-1);                                                            \
        }                                                                        \
    } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

namespace cupti_tracer {

class CUptiManager;

class Tracer {
    friend class CUptiManager;
    public:
        struct Record {
            const char *kind;
            const char *name;
            uint64_t start, end; // start & end time of the record in ns
            uint32_t processId, threadId; // deviceID & streamID for a kernel record
            uint32_t correlationId;
        };

        Tracer();
        void start();
        void stop();
        void print_trace();
        std::vector<Record> get_records() {
            return records_;
        }

    private:
        static void CUPTIAPI APICallback(void *userdata, CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid, const void *cbdata);

        void ActivityCallback(const CUpti_Activity &record);

    private:
        CUpti_SubscriberHandle subscriber_;
        std::vector<Record> records_;
        CUptiManager *manager_;
}; // end of class Tracer

std::shared_ptr<Tracer> _GetSharedRef() {
    static std::shared_ptr<Tracer> inst(new Tracer());
    return inst;
}

Tracer* Get() {
    static Tracer *ptr = _GetSharedRef().get();
    return ptr;
}

class CUptiManager {
    public:
        static CUptiManager *Get();
        void Initialize(Tracer *tracer);

    private:
        static void BufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
            Get()->InternalBufferRequested(buffer, size, maxNumRecords);
        }

        static void BufferCompleted(CUcontext ctx, uint32_t streamId,
                                    uint8_t *buffer, size_t size, size_t validSize) {
            Get()->InternalBufferCompleted(ctx, streamId, buffer, size, validSize);
        }

        void InternalBufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
        void InternalBufferCompleted(CUcontext ctx, uint32_t streamId,
                                     uint8_t *buffer, size_t size, size_t validSize);

        Tracer *tracer_;
}; // end of class CUptiManager


void CUptiManager::InternalBufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    printf("InternalBufferRequested\n");
    uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
    if (bfr == NULL) {
        printf("Error: out of memory\n");
        exit(-1);
    }

    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
}

void CUptiManager::InternalBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size, size_t validSize) {
    CUpti_Activity *record = nullptr;
    fprintf(stderr, "InternalBufferCompleted\n");
    if (validSize > 0) {
        do {
            auto status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                tracer_->ActivityCallback(*record);
            }
            else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            else {
                CUPTI_CALL(status);
            }
        } while (1);
    }
    free(buffer);
}

void CUptiManager::Initialize(Tracer *tracer) {
    tracer_ = tracer;
    cuptiActivityRegisterCallbacks((CUpti_BuffersCallbackRequestFunc)BufferRequested,
                                   (CUpti_BuffersCallbackCompleteFunc)BufferCompleted);
}

CUptiManager *CUptiManager::Get() {
    static auto manager = new CUptiManager();
    printf("%p\n", manager);
    return manager;
}



Tracer::Tracer() {
    manager_ = CUptiManager::Get();
    manager_->Initialize(this);
}

void Tracer::start() {
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
    CUPTI_CALL(cuptiSubscribe(&subscriber_, static_cast<CUpti_CallbackFunc>(APICallback), this));
    CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                                   CUPTI_CB_DOMAIN_DRIVER_API,
                                   CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                                   CUPTI_CB_DOMAIN_DRIVER_API,
                                   CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz));
    CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                                   CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
}

void Tracer::stop() {
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONTEXT));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_NAME));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MARKER));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD));
    CUPTI_CALL(cuptiUnsubscribe(subscriber_));

    //CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
    //CUPTI_CALL(cuptiEnableCallback(/*disable=*/1, subscriber_,
    //                               CUPTI_CB_DOMAIN_DRIVER_API,
    //                               CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    CUPTI_CALL(cuptiActivityFlushAll(0));
}

void Tracer::APICallback(void *userdata, CUpti_CallbackDomain domain,
                         CUpti_CallbackId cbid, const void *cbdata) {
    auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
    Tracer *tracer = reinterpret_cast<Tracer *>(userdata);
    //std::cout << cbInfo->context << std::endl;
    //std::cout << cbInfo->contextUid << std::endl;
    //std::cout << cbInfo->correlationId << std::endl;
    //std::cout << cbInfo->functionName << std::endl;
    //std::cout << cbInfo->symbolName << std::endl;
    switch (domain) {
        case CUPTI_CB_DOMAIN_RUNTIME_API:
            //std::cout << cbid << std::endl;
            //std::cout << cbInfo->correlationId << std::endl;
            //std::cout << cbInfo->functionName << std::endl;
            //std::cout << cbInfo->symbolName << std::endl;
            break;
        case CUPTI_CB_DOMAIN_DRIVER_API:
            break;
    }
}

void Tracer::ActivityCallback(const CUpti_Activity &record) {
    switch (record.kind) {
        case CUPTI_ACTIVITY_KIND_DRIVER: {
            auto *api = reinterpret_cast<const CUpti_ActivityAPI *>(&record);
            records_.push_back(Record{"DRIVER", "",
                               (unsigned long long) (api->start),
                               (unsigned long long) (api->end),
                               api->processId, api->threadId, api->correlationId});
            fprintf(stderr, "DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u, duration %u\n",
                api->cbid,
                (unsigned long long) (api->start),
                (unsigned long long) (api->end),
                api->processId, api->threadId, api->correlationId,
                (unsigned long long) (api->end - api->start));
                break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
            auto *api = reinterpret_cast<const CUpti_ActivityAPI *>(&record);
            records_.push_back(Record{"RUNTIME", "",
                               (unsigned long long) (api->start),
                               (unsigned long long) (api->end),
                               api->processId, api->threadId, api->correlationId});
            fprintf(stderr, "RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u, duration %u\n",
                api->cbid,
                (unsigned long long) (api->start),
                (unsigned long long) (api->end),
                api->processId, api->threadId, api->correlationId,
                (unsigned long long) (api->end - api->start));
                break;
        }
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            auto *kernel = reinterpret_cast<const CUpti_ActivityKernel4 *>(&record);
            const char* kindString = (record.kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
            records_.push_back(Record{kindString, kernel->name,
                               (unsigned long long) (kernel->start),
                               (unsigned long long) (kernel->end),
                               kernel->deviceId, kernel->streamId, kernel->correlationId});

            fprintf(stderr, "%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u, duration %u\n",
                kindString,
                kernel->name,
                (unsigned long long) (kernel->start),
                (unsigned long long) (kernel->end),
                kernel->deviceId, kernel->contextId, kernel->streamId,
                kernel->correlationId,
                (unsigned long long) (kernel->end - kernel->start));
            fprintf(stderr, "    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
                kernel->gridX, kernel->gridY, kernel->gridZ,
                kernel->blockX, kernel->blockY, kernel->blockZ,
                kernel->staticSharedMemory, kernel->dynamicSharedMemory);
            break;
        }
    }
}

void Tracer::print_trace() {
    fprintf(stderr, "%lu records\n", records_.size());
    for (int i = 0; i < records_.size(); ++ i) {
        auto record = records_[i];
        fprintf(stderr, "%s %s [%u - %u]", record.kind, record.name, record.start, record.end);
    }
}

} // end of namespace cupti_tracer

static PyObject * THPModule_startCUptiTracing(PyObject *_unused) {
    static auto tracer = cupti_tracer::Get();
    tracer->start();
    Py_RETURN_NONE;
}

static PyObject * THPModule_endCUptiTracing(PyObject *_unused) {
    static auto tracer = cupti_tracer::Get();
    tracer->stop();
    auto records = tracer->get_records();
    PyObject* ret = PyList_New(records.size());
    for (int i = 0; i < records.size(); ++ i) {
        auto record = records[i];
        PyObject* py_record = PyList_New(7);
        PyList_SetItem(py_record, 0, PyUnicode_FromString(record.kind));
        PyList_SetItem(py_record, 1, PyUnicode_FromString(record.name));
        PyList_SetItem(py_record, 2, PyLong_FromUnsignedLongLong(record.start));
        PyList_SetItem(py_record, 3, PyLong_FromUnsignedLongLong(record.end));
        PyList_SetItem(py_record, 4, PyLong_FromUnsignedLong(record.processId));
        PyList_SetItem(py_record, 5, PyLong_FromUnsignedLong(record.threadId));
        PyList_SetItem(py_record, 6, PyLong_FromUnsignedLong(record.correlationId));
        PyList_SetItem(ret, i, py_record);
    }
    return ret;
    //Py_RETURN_NONE;
}

PyMethodDef CUptiMethods[] = {
    {"start_cupti_tracing", (PyCFunction)THPModule_startCUptiTracing, METH_NOARGS, NULL},
    {"end_cupti_tracing",   (PyCFunction)THPModule_endCUptiTracing,   METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};
