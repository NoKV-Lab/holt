#ifndef HOLT_FFI_H
#define HOLT_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HoltTree HoltTree;
typedef struct HoltIter HoltIter;

enum {
  HOLT_OK = 0,
  HOLT_ITER_END = 1,
  HOLT_ERR = -1
};

enum {
  HOLT_ENTRY_KEY = 1,
  HOLT_ENTRY_COMMON_PREFIX = 2
};

enum {
  HOLT_ATOMIC_PUT = 1,
  HOLT_ATOMIC_DELETE = 2
};

typedef struct HoltBytes {
  uint8_t *ptr;
  size_t len;
} HoltBytes;

typedef struct HoltRecord {
  uint8_t found;
  HoltBytes value;
  uint64_t version;
} HoltRecord;

typedef struct HoltEntry {
  uint32_t kind;
  HoltBytes path;
  HoltBytes value;
  uint64_t version;
} HoltEntry;

typedef struct HoltAtomicOp {
  uint32_t kind;
  const uint8_t *key;
  size_t key_len;
  const uint8_t *value;
  size_t value_len;
} HoltAtomicOp;

const char *holt_last_error_message(void);

int32_t holt_tree_open_with_wal_sync(const char *path, uint8_t wal_sync, HoltTree **out);
int32_t holt_tree_open_memory(HoltTree **out);
void holt_tree_close(HoltTree *tree);

int32_t holt_tree_put(HoltTree *tree, const uint8_t *key, size_t key_len,
                      const uint8_t *value, size_t value_len);
int32_t holt_tree_get(HoltTree *tree, const uint8_t *key, size_t key_len,
                      HoltRecord *out);
int32_t holt_tree_delete(HoltTree *tree, const uint8_t *key, size_t key_len,
                         uint8_t *existed_out);
int32_t holt_tree_atomic(HoltTree *tree, const HoltAtomicOp *ops, size_t op_count,
                         uint8_t *committed_out);
int32_t holt_tree_checkpoint(HoltTree *tree);

int32_t holt_tree_scan_keys(HoltTree *tree, const uint8_t *prefix, size_t prefix_len,
                            int32_t delimiter, const uint8_t *start_after,
                            size_t start_after_len, HoltIter **out);
int32_t holt_tree_scan_records(HoltTree *tree, const uint8_t *prefix, size_t prefix_len,
                               int32_t delimiter, const uint8_t *start_after,
                               size_t start_after_len, HoltIter **out);

int32_t holt_iter_next(HoltIter *iter, HoltEntry *out);
void holt_iter_close(HoltIter *iter);

void holt_bytes_free(HoltBytes bytes);
void holt_record_free(HoltRecord *record);
void holt_entry_free(HoltEntry *entry);

#ifdef __cplusplus
}
#endif

#endif
