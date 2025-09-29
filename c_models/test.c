#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void entry(const float tensor_images[1][3][640][640], float tensor_output0[1][84][8400]);

int load_ppm(const char* filename, unsigned char** data, int* width, int* height) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;

    char format[3];
    if (fscanf(f, "%2s", format) != 1) { fclose(f); return 0; }
    if (strcmp(format, "P6") != 0) { fclose(f); return 0; }

    if (fscanf(f, "%d %d", width, height) != 2) { fclose(f); return 0; }
    int maxval;
    if (fscanf(f, "%d", &maxval) != 1) { fclose(f); return 0; }
    fgetc(f);

    *data = malloc((*width) * (*height) * 3);
    if (!*data) { fclose(f); return 0; }
    size_t nread = fread(*data, 1, (*width) * (*height) * 3, f);
    fclose(f);
    if (nread != (size_t)((*width) * (*height) * 3)) {
        free(*data);
        *data = NULL;
        return 0;
    }
    return 1;
}

void preprocess(unsigned char* img, int w, int h, float input[1][3][640][640]) {
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 640; y++) {
            for (int x = 0; x < 640; x++) {
                int orig_x = (x * w) / 640;
                int orig_y = (y * h) / 640;
                if (orig_x >= w) orig_x = w - 1;
                if (orig_y >= h) orig_y = h - 1;
                
                unsigned char pixel = img[(orig_y * w + orig_x) * 3 + c];
                input[0][c][y][x] = pixel / 255.0f;
            }
        }
    }
}

int main() {
    printf("=== YOLOv8 C Model Test ===\n");
    
    static float input[1][3][640][640];
    static float output[1][84][8400];
    
    unsigned char* img_data;
    int width, height;
    const char* img_path = "../images/street.ppm";
    
    if (load_ppm(img_path, &img_data, &width, &height)) {
        printf("Loaded image: %dx%d\n", width, height);
        preprocess(img_data, width, height, input);
        free(img_data);
    } else {
        printf("Using synthetic data\n");
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < 640; y++) {
                for (int x = 0; x < 640; x++) {
                    input[0][c][y][x] = (float)(x + y + c * 100) / 1000.0f;
                }
            }
        }
    }
    
    printf("Running inference...\n");
    clock_t start = clock();
    entry(input, output);
    clock_t end = clock();
    
    double time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Inference time: %.2f ms\n", time_ms);
    

    // Set the confidence threshold here
    const float CONFIDENCE_THRESHOLD = 0.5f;
    printf("Detection confidence threshold: %.2f\n", CONFIDENCE_THRESHOLD);


    // --- Simple NMS implementation ---
    typedef struct {
        float x, y, w, h, conf;
        int class_id;
        int suppressed;
    } Detection;

    Detection dets[8400];
    int det_count = 0;

    // Collect all detections above threshold
    for (int i = 0; i < 8400; i++) {
        float x_center = output[0][0][i];
        float y_center = output[0][1][i];
        float width = output[0][2][i];
        float height = output[0][3][i];
        float max_conf = 0.0f;
        int best_class = -1;
        for (int c = 4; c < 84; c++) {
            float class_conf = output[0][c][i];
            if (class_conf > max_conf) {
                max_conf = class_conf;
                best_class = c - 4;
            }
        }
        if (max_conf > CONFIDENCE_THRESHOLD) {
            dets[det_count].x = x_center;
            dets[det_count].y = y_center;
            dets[det_count].w = width;
            dets[det_count].h = height;
            dets[det_count].conf = max_conf;
            dets[det_count].class_id = best_class;
            dets[det_count].suppressed = 0;
            det_count++;
        }
    }

    // Helper: compute IoU between two boxes (center format)
    float iou(Detection* a, Detection* b) {
        float x1 = a->x - a->w/2, y1 = a->y - a->h/2;
        float x2 = a->x + a->w/2, y2 = a->y + a->h/2;
        float x1b = b->x - b->w/2, y1b = b->y - b->h/2;
        float x2b = b->x + b->w/2, y2b = b->y + b->h/2;
        float xx1 = (x1 > x1b) ? x1 : x1b;
        float yy1 = (y1 > y1b) ? y1 : y1b;
        float xx2 = (x2 < x2b) ? x2 : x2b;
        float yy2 = (y2 < y2b) ? y2 : y2b;
        float w = xx2 - xx1;
        float h = yy2 - yy1;
        if (w <= 0 || h <= 0) return 0.0f;
        float inter = w * h;
        float area_a = (x2-x1)*(y2-y1);
        float area_b = (x2b-x1b)*(y2b-y1b);
        return inter / (area_a + area_b - inter);
    }

    // NMS: for each class, suppress boxes with high IoU
    float NMS_IOU_THRESH = 0.5f;
    for (int c = 0; c < 80; c++) {
        // For each detection of this class
        for (int i = 0; i < det_count; i++) {
            if (dets[i].class_id != c || dets[i].suppressed) continue;
            for (int j = i+1; j < det_count; j++) {
                if (dets[j].class_id != c || dets[j].suppressed) continue;
                if (iou(&dets[i], &dets[j]) > NMS_IOU_THRESH) {
                    // Suppress lower-confidence box
                    if (dets[i].conf >= dets[j].conf)
                        dets[j].suppressed = 1;
                    else
                        dets[i].suppressed = 1;
                }
            }
        }
    }

    FILE* f = fopen("../c_detections.txt", "w");
    if (f) {
        fprintf(f, "YOLOv8 Detections (x_center y_center width height confidence class)\n");
        fprintf(f, "Confidence threshold: %.2f\n", CONFIDENCE_THRESHOLD);
        fprintf(f, "NMS IoU threshold: %.2f\n", NMS_IOU_THRESH);

        int valid_detections = 0;
        for (int i = 0; i < det_count && valid_detections < 20; i++) {
            if (!dets[i].suppressed) {
                fprintf(f, "Det %d: %.1f %.1f %.1f %.1f %.3f %d\n",
                        valid_detections, dets[i].x, dets[i].y, dets[i].w, dets[i].h, dets[i].conf, dets[i].class_id);
                valid_detections++;
            }
        }

        if (valid_detections == 0) {
            fprintf(f, "No detections above confidence threshold %.2f\n", CONFIDENCE_THRESHOLD);
            // Show some raw values for debugging
            fprintf(f, "Raw values (first 5 boxes):\n");
            for (int i = 0; i < 5 && i < det_count; i++) {
                fprintf(f, "Box %d: x=%.3f y=%.3f w=%.3f h=%.3f conf=%.3f class=%d\n", i,
                        dets[i].x, dets[i].y, dets[i].w, dets[i].h, dets[i].conf, dets[i].class_id);
            }
        }

        fclose(f);
        printf("Found %d valid detections after NMS (confidence >= %.2f)\n", valid_detections, CONFIDENCE_THRESHOLD);
    }
    
    printf("Test completed!\n");
    return 0;
}
