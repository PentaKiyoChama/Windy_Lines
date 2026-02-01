#include <stdio.h>
#include <math.h>

// Test the OutInCirc bezier curve (NEW VERSION)
float OutInCircBezier(float t) {
    const float p1y = 0.9f;
    const float p2y = 0.9f;
    const float u = 1.0f - t;
    const float u2 = u * u;
    const float t2 = t * t;
    return u2 * u * 0.0f + 3.0f * u2 * t * p1y + 3.0f * u * t2 * p2y + t2 * t * 1.0f;
}

// Calculate derivative (velocity)
float GetVelocity(float t) {
    const float epsilon = 0.001f;
    const float t1 = t > epsilon ? t - epsilon : 0.0f;
    const float t2 = t < 1.0f - epsilon ? t + epsilon : 1.0f;
    const float dt = t2 - t1;
    if (dt > 0.0f) {
        return (OutInCircBezier(t2) - OutInCircBezier(t1)) / dt;
    }
    return 1.0f;
}

int main() {
    printf("OutInCirc Bezier Curve Test - P0(0,0), P1(0.15,0.9), P2(0.85,0.9), P3(1,1)\n");
    printf("========================================================================\n");
    printf("t     | Position | Velocity | Analysis\n");
    printf("------|----------|----------|----------------------------------\n");
    
    for (float t = 0.0f; t <= 1.0f; t += 0.1f) {
        float pos = OutInCircBezier(t);
        float vel = GetVelocity(t);
        
        const char* analysis = "";
        if (t < 0.15f) analysis = "Fast rise (should be high velocity)";
        else if (t < 0.85f) analysis = "Plateau (should be low velocity)";
        else analysis = "Fast fall (should be high velocity)";
        
        printf("%.2f  | %.4f   | %.4f   | %s\n", t, pos, vel, analysis);
    }
    
    printf("\n=== VERIFICATION ===\n");
    printf("Expected behavior:\n");
    printf("- t=0.0→0.2:  Fast rise (velocity > 1.5)\n");
    printf("- t=0.3→0.7:  Plateau (velocity ≈ 0.1-0.3)\n");
    printf("- t=0.8→1.0:  Fast fall (velocity > 1.5)\n");
    printf("- Position should go from 0→1 smoothly\n");
    
    // Check specific points
    printf("\n=== SPECIFIC CHECKS ===\n");
    printf("t=0.5 position: %.4f (should be ~0.95, at plateau)\n", OutInCircBezier(0.5f));
    printf("t=0.5 velocity: %.4f (should be ~0.1, very slow)\n", GetVelocity(0.5f));
    printf("t=0.1 velocity: %.4f (should be >1.5, rising fast)\n", GetVelocity(0.1f));
    printf("t=0.9 velocity: %.4f (should be >1.5, falling fast)\n", GetVelocity(0.9f));
    
    return 0;
}
