/**
 * Test validation for parameter reordering system
 * 
 * This test verifies that:
 * 1. All parameters are included exactly once in PARAM_DISPLAY_ORDER
 * 2. SDK_PROCAMP_INPUT is first
 * 3. Topic groups are properly formed (basic structural check)
 * 4. GetParamDisplayIndex and GetParamIdAtDisplayIndex work correctly
 * 
 * Build and run this test to validate the parameter order system.
 * 
 * NOTE: This is a standalone test that doesn't require Adobe SDK headers.
 * It includes only the enum definitions and PARAM_DISPLAY_ORDER array.
 */

#include <stdio.h>
#include <stdlib.h>
#include <set>

// Simplified enum definitions (from SDK_ProcAmp.h)
enum
{
	SDK_PROCAMP_INPUT = 0,
	SDK_PROCAMP_EFFECT_PRESET,
	SDK_PROCAMP_LINE_SEED,
	SDK_PROCAMP_LINE_COUNT,
	SDK_PROCAMP_LINE_LIFETIME,
	SDK_PROCAMP_LINE_INTERVAL,
	SDK_PROCAMP_LINE_TRAVEL,
	SDK_PROCAMP_LINE_EASING,
	SDK_PROCAMP_COLOR_MODE,
	SDK_PROCAMP_LINE_COLOR,
	SDK_PROCAMP_COLOR_PRESET,
	SDK_PROCAMP_CUSTOM_COLOR_1,
	SDK_PROCAMP_CUSTOM_COLOR_2,
	SDK_PROCAMP_CUSTOM_COLOR_3,
	SDK_PROCAMP_CUSTOM_COLOR_4,
	SDK_PROCAMP_CUSTOM_COLOR_5,
	SDK_PROCAMP_CUSTOM_COLOR_6,
	SDK_PROCAMP_CUSTOM_COLOR_7,
	SDK_PROCAMP_CUSTOM_COLOR_8,
	SDK_PROCAMP_LINE_THICKNESS,
	SDK_PROCAMP_LINE_LENGTH,
	SDK_PROCAMP_LINE_ANGLE,
	SDK_PROCAMP_LINE_CAP,
	SDK_PROCAMP_LINE_TAIL_FADE,
	SDK_PROCAMP_POSITION_HEADER,
	SDK_PROCAMP_SPAWN_SOURCE,
	SDK_PROCAMP_LINE_ALPHA_THRESH,
	SDK_PROCAMP_LINE_ORIGIN_MODE,
	SDK_PROCAMP_ANIM_PATTERN,
	SDK_PROCAMP_LINE_START_TIME,
	SDK_PROCAMP_LINE_DURATION,
	SDK_PROCAMP_LINE_DEPTH_STRENGTH,
	SDK_PROCAMP_CENTER_GAP,
	SDK_PROCAMP_ORIGIN_OFFSET_X,
	SDK_PROCAMP_ORIGIN_OFFSET_Y,
	SDK_PROCAMP_LINE_SPAWN_SCALE_X,
	SDK_PROCAMP_LINE_SPAWN_SCALE_Y,
	SDK_PROCAMP_LINE_SPAWN_ROTATION,
	SDK_PROCAMP_LINE_SHOW_SPAWN_AREA,
	SDK_PROCAMP_LINE_SPAWN_AREA_COLOR,
	SDK_PROCAMP_POSITION_TOPIC_END,
	SDK_PROCAMP_SHADOW_HEADER,
	SDK_PROCAMP_SHADOW_ENABLE,
	SDK_PROCAMP_SHADOW_COLOR,
	SDK_PROCAMP_SHADOW_OFFSET_X,
	SDK_PROCAMP_SHADOW_OFFSET_Y,
	SDK_PROCAMP_SHADOW_OPACITY,
	SDK_PROCAMP_SHADOW_TOPIC_END,
	SDK_PROCAMP_MOTION_BLUR_HEADER,
	SDK_PROCAMP_MOTION_BLUR_ENABLE,
	SDK_PROCAMP_MOTION_BLUR_SAMPLES,
	SDK_PROCAMP_MOTION_BLUR_STRENGTH,
	SDK_PROCAMP_MOTION_BLUR_TOPIC_END,
	SDK_PROCAMP_ADVANCED_HEADER,
	SDK_PROCAMP_LINE_AA,
	SDK_PROCAMP_HIDE_ELEMENT,
	SDK_PROCAMP_BLEND_MODE,
	SDK_PROCAMP_ADVANCED_TOPIC_END,
	SDK_PROCAMP_LINKAGE_HEADER,
	SDK_PROCAMP_LENGTH_LINKAGE,
	SDK_PROCAMP_LENGTH_LINKAGE_RATE,
	SDK_PROCAMP_THICKNESS_LINKAGE,
	SDK_PROCAMP_THICKNESS_LINKAGE_RATE,
	SDK_PROCAMP_TRAVEL_LINKAGE,
	SDK_PROCAMP_TRAVEL_LINKAGE_RATE,
	SDK_PROCAMP_LINKAGE_TOPIC_END,
	SDK_PROCAMP_LINE_ALLOW_MIDPLAY,
	SDK_PROCAMP_LINE_COLOR_R,
	SDK_PROCAMP_LINE_COLOR_G,
	SDK_PROCAMP_LINE_COLOR_B,
	SDK_PROCAMP_NUM_PARAMS
};

// Display order array (from SDK_ProcAmp_ParamOrder.h)
static const int PARAM_DISPLAY_ORDER[] = {
	SDK_PROCAMP_INPUT,
	SDK_PROCAMP_EFFECT_PRESET,
	SDK_PROCAMP_LINE_SEED,
	SDK_PROCAMP_LINE_COUNT,
	SDK_PROCAMP_LINE_LIFETIME,
	SDK_PROCAMP_LINE_INTERVAL,
	SDK_PROCAMP_LINE_TRAVEL,
	SDK_PROCAMP_LINE_EASING,
	SDK_PROCAMP_COLOR_MODE,
	SDK_PROCAMP_LINE_COLOR,
	SDK_PROCAMP_COLOR_PRESET,
	SDK_PROCAMP_CUSTOM_COLOR_1,
	SDK_PROCAMP_CUSTOM_COLOR_2,
	SDK_PROCAMP_CUSTOM_COLOR_3,
	SDK_PROCAMP_CUSTOM_COLOR_4,
	SDK_PROCAMP_CUSTOM_COLOR_5,
	SDK_PROCAMP_CUSTOM_COLOR_6,
	SDK_PROCAMP_CUSTOM_COLOR_7,
	SDK_PROCAMP_CUSTOM_COLOR_8,
	SDK_PROCAMP_LINE_THICKNESS,
	SDK_PROCAMP_LINE_LENGTH,
	SDK_PROCAMP_LINE_ANGLE,
	SDK_PROCAMP_LINE_CAP,
	SDK_PROCAMP_LINE_TAIL_FADE,
	SDK_PROCAMP_POSITION_HEADER,
	SDK_PROCAMP_SPAWN_SOURCE,
	SDK_PROCAMP_LINE_ALPHA_THRESH,
	SDK_PROCAMP_LINE_ORIGIN_MODE,
	SDK_PROCAMP_ANIM_PATTERN,
	SDK_PROCAMP_LINE_START_TIME,
	SDK_PROCAMP_LINE_DURATION,
	SDK_PROCAMP_LINE_DEPTH_STRENGTH,
	SDK_PROCAMP_CENTER_GAP,
	SDK_PROCAMP_ORIGIN_OFFSET_X,
	SDK_PROCAMP_ORIGIN_OFFSET_Y,
	SDK_PROCAMP_LINE_SPAWN_SCALE_X,
	SDK_PROCAMP_LINE_SPAWN_SCALE_Y,
	SDK_PROCAMP_LINE_SPAWN_ROTATION,
	SDK_PROCAMP_LINE_SHOW_SPAWN_AREA,
	SDK_PROCAMP_LINE_SPAWN_AREA_COLOR,
	SDK_PROCAMP_POSITION_TOPIC_END,
	SDK_PROCAMP_SHADOW_HEADER,
	SDK_PROCAMP_SHADOW_ENABLE,
	SDK_PROCAMP_SHADOW_COLOR,
	SDK_PROCAMP_SHADOW_OFFSET_X,
	SDK_PROCAMP_SHADOW_OFFSET_Y,
	SDK_PROCAMP_SHADOW_OPACITY,
	SDK_PROCAMP_SHADOW_TOPIC_END,
	SDK_PROCAMP_MOTION_BLUR_HEADER,
	SDK_PROCAMP_MOTION_BLUR_ENABLE,
	SDK_PROCAMP_MOTION_BLUR_SAMPLES,
	SDK_PROCAMP_MOTION_BLUR_STRENGTH,
	SDK_PROCAMP_MOTION_BLUR_TOPIC_END,
	SDK_PROCAMP_ADVANCED_HEADER,
	SDK_PROCAMP_LINE_AA,
	SDK_PROCAMP_HIDE_ELEMENT,
	SDK_PROCAMP_BLEND_MODE,
	SDK_PROCAMP_ADVANCED_TOPIC_END,
	SDK_PROCAMP_LINKAGE_HEADER,
	SDK_PROCAMP_LENGTH_LINKAGE,
	SDK_PROCAMP_LENGTH_LINKAGE_RATE,
	SDK_PROCAMP_THICKNESS_LINKAGE,
	SDK_PROCAMP_THICKNESS_LINKAGE_RATE,
	SDK_PROCAMP_TRAVEL_LINKAGE,
	SDK_PROCAMP_TRAVEL_LINKAGE_RATE,
	SDK_PROCAMP_LINKAGE_TOPIC_END,
	SDK_PROCAMP_LINE_ALLOW_MIDPLAY,
	SDK_PROCAMP_LINE_COLOR_R,
	SDK_PROCAMP_LINE_COLOR_G,
	SDK_PROCAMP_LINE_COLOR_B,
};

// Helper functions
inline int GetParamDisplayIndex(int paramId)
{
	for (int i = 0; i < SDK_PROCAMP_NUM_PARAMS; ++i)
	{
		if (PARAM_DISPLAY_ORDER[i] == paramId)
		{
			return i;
		}
	}
	return -1;
}

inline int GetParamIdAtDisplayIndex(int displayIndex)
{
	if (displayIndex >= 0 && displayIndex < SDK_PROCAMP_NUM_PARAMS)
	{
		return PARAM_DISPLAY_ORDER[displayIndex];
	}
	return -1;
}

// Test result tracking
static int test_passed = 0;
static int test_failed = 0;

#define TEST(condition, message) \
	do { \
		if (condition) { \
			printf("[PASS] %s\n", message); \
			test_passed++; \
		} else { \
			printf("[FAIL] %s\n", message); \
			test_failed++; \
		} \
	} while(0)

int main()
{
	printf("==============================================\n");
	printf("Parameter Reordering System Validation Test\n");
	printf("==============================================\n\n");
	
	// Test 1: Array size matches parameter count
	TEST(sizeof(PARAM_DISPLAY_ORDER) / sizeof(PARAM_DISPLAY_ORDER[0]) == SDK_PROCAMP_NUM_PARAMS,
	     "Array size matches SDK_PROCAMP_NUM_PARAMS");
	
	// Test 2: SDK_PROCAMP_INPUT is first
	TEST(PARAM_DISPLAY_ORDER[0] == SDK_PROCAMP_INPUT,
	     "SDK_PROCAMP_INPUT is at index 0");
	
	// Test 3: All parameter IDs are included exactly once
	std::set<int> paramIds;
	bool hasDuplicates = false;
	for (int i = 0; i < SDK_PROCAMP_NUM_PARAMS; i++)
	{
		int paramId = PARAM_DISPLAY_ORDER[i];
		if (paramIds.count(paramId) > 0)
		{
			printf("[ERROR] Duplicate parameter ID %d found at index %d\n", paramId, i);
			hasDuplicates = true;
		}
		paramIds.insert(paramId);
	}
	TEST(!hasDuplicates, "No duplicate parameter IDs");
	
	// Test 4: All parameter IDs are in valid range
	bool allInRange = true;
	for (int i = 0; i < SDK_PROCAMP_NUM_PARAMS; i++)
	{
		int paramId = PARAM_DISPLAY_ORDER[i];
		if (paramId < 0 || paramId >= SDK_PROCAMP_NUM_PARAMS)
		{
			printf("[ERROR] Parameter ID %d at index %d is out of range\n", paramId, i);
			allInRange = false;
		}
	}
	TEST(allInRange, "All parameter IDs in valid range [0, SDK_PROCAMP_NUM_PARAMS)");
	
	// Test 5: GetParamDisplayIndex returns correct values
	bool displayIndexCorrect = true;
	for (int i = 0; i < SDK_PROCAMP_NUM_PARAMS; i++)
	{
		int paramId = PARAM_DISPLAY_ORDER[i];
		int foundIndex = GetParamDisplayIndex(paramId);
		if (foundIndex != i)
		{
			printf("[ERROR] GetParamDisplayIndex(%d) returned %d, expected %d\n", 
			       paramId, foundIndex, i);
			displayIndexCorrect = false;
		}
	}
	TEST(displayIndexCorrect, "GetParamDisplayIndex returns correct indices");
	
	// Test 6: GetParamIdAtDisplayIndex returns correct values
	bool idAtIndexCorrect = true;
	for (int i = 0; i < SDK_PROCAMP_NUM_PARAMS; i++)
	{
		int expectedId = PARAM_DISPLAY_ORDER[i];
		int actualId = GetParamIdAtDisplayIndex(i);
		if (actualId != expectedId)
		{
			printf("[ERROR] GetParamIdAtDisplayIndex(%d) returned %d, expected %d\n",
			       i, actualId, expectedId);
			idAtIndexCorrect = false;
		}
	}
	TEST(idAtIndexCorrect, "GetParamIdAtDisplayIndex returns correct IDs");
	
	// Test 7: Topic headers have corresponding endings
	// This is a basic check - just verifies we have equal number of headers and endings
	int topicHeaders = 0;
	int topicEndings = 0;
	for (int i = 0; i < SDK_PROCAMP_NUM_PARAMS; i++)
	{
		int paramId = PARAM_DISPLAY_ORDER[i];
		
		// Check if it's a header (naming convention: *_HEADER)
		if (paramId == SDK_PROCAMP_POSITION_HEADER ||
		    paramId == SDK_PROCAMP_SHADOW_HEADER ||
		    paramId == SDK_PROCAMP_MOTION_BLUR_HEADER ||
		    paramId == SDK_PROCAMP_ADVANCED_HEADER ||
		    paramId == SDK_PROCAMP_LINKAGE_HEADER)
		{
			topicHeaders++;
		}
		
		// Check if it's an ending (naming convention: *_TOPIC_END)
		if (paramId == SDK_PROCAMP_POSITION_TOPIC_END ||
		    paramId == SDK_PROCAMP_SHADOW_TOPIC_END ||
		    paramId == SDK_PROCAMP_MOTION_BLUR_TOPIC_END ||
		    paramId == SDK_PROCAMP_ADVANCED_TOPIC_END ||
		    paramId == SDK_PROCAMP_LINKAGE_TOPIC_END)
		{
			topicEndings++;
		}
	}
	TEST(topicHeaders == topicEndings && topicHeaders == 5,
	     "Topic headers and endings are balanced (5 groups)");
	
	// Test 8: Topic structure validation
	// Verifies that each topic group has proper nesting (HEADER before TOPIC_END)
	struct TopicPair {
		int headerId;
		int endId;
		const char* name;
	};
	
	TopicPair topics[] = {
		{ SDK_PROCAMP_POSITION_HEADER, SDK_PROCAMP_POSITION_TOPIC_END, "Position" },
		{ SDK_PROCAMP_SHADOW_HEADER, SDK_PROCAMP_SHADOW_TOPIC_END, "Shadow" },
		{ SDK_PROCAMP_MOTION_BLUR_HEADER, SDK_PROCAMP_MOTION_BLUR_TOPIC_END, "Motion Blur" },
		{ SDK_PROCAMP_ADVANCED_HEADER, SDK_PROCAMP_ADVANCED_TOPIC_END, "Advanced" },
		{ SDK_PROCAMP_LINKAGE_HEADER, SDK_PROCAMP_LINKAGE_TOPIC_END, "Linkage" }
	};
	
	bool topicStructureValid = true;
	for (int t = 0; t < 5; t++)
	{
		int headerIdx = GetParamDisplayIndex(topics[t].headerId);
		int endIdx = GetParamDisplayIndex(topics[t].endId);
		
		if (headerIdx >= endIdx)
		{
			printf("[ERROR] Topic '%s': HEADER (idx %d) not before TOPIC_END (idx %d)\n",
			       topics[t].name, headerIdx, endIdx);
			topicStructureValid = false;
		}
		
		if (endIdx - headerIdx < 2)
		{
			printf("[WARNING] Topic '%s': No parameters between HEADER and TOPIC_END\n",
			       topics[t].name);
		}
	}
	TEST(topicStructureValid, "All topic groups have HEADER before TOPIC_END");
	
	// Print summary
	printf("\n==============================================\n");
	printf("Test Summary\n");
	printf("==============================================\n");
	printf("Passed: %d\n", test_passed);
	printf("Failed: %d\n", test_failed);
	printf("Total:  %d\n", test_passed + test_failed);
	printf("\n");
	
	if (test_failed == 0)
	{
		printf("✅ All tests passed! Parameter reordering system is valid.\n");
		return 0;
	}
	else
	{
		printf("❌ Some tests failed. Please review PARAM_DISPLAY_ORDER.\n");
		return 1;
	}
}
