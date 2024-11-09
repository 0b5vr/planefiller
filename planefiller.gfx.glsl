#version 430

layout(binding = 0) uniform sampler2D b0;
layout(binding = 1) uniform sampler2D b1;
layout(location = 0) uniform int waveOutPosition;
#if defined(EXPORT_EXECUTABLE)
  vec2 resolution = {SCREEN_XRESO, SCREEN_YRESO};
  #define NUM_SAMPLES_PER_SEC 48000.
  float globalTime = waveOutPosition / NUM_SAMPLES_PER_SEC;
#else
  layout(location = 2) uniform float globalTime;
  layout(location = 3) uniform vec2 resolution;
#endif

#if defined(EXPORT_EXECUTABLE)
  #pragma work_around_begin:layout(std430,binding=0)buffer _{vec2 %s[];};
  vec2 waveOutSamples[];
  #pragma work_around_end
#else
  layout(std430, binding = 0) buffer _{ vec2 waveOutSamples[]; };
#endif

layout(location = 0) out vec4 outColor0;
layout(location = 1) out vec4 outColor1;

// == constants ====================================================================================
const float FAR = 44.0;

const float PI = acos(-1.0);
const float TAU = PI + PI;
const float SQRT2 = sqrt(2.0);
const float SQRT3 = sqrt(3.0);
const float SQRT3_OVER_TWO = SQRT3 / 2.0;

const float i_B2T = 0.43;
const float i_SWING = 0.62;
const int i_SAMPLES = 20;
const float i_SAMPLES_F = 20.0;
const int i_PLANES_TRAVERSAL = 16;
const int i_REFLECTS = 3;

const float i_GREEBLES_GAP = 0.01;
const int i_GREEBLES_TRAVERSAL = 8;
const float i_GREEBLES_HEIGHT = 0.03;
const float i_PLANE_INTERVAL = 0.5;

// == macros =======================================================================================
#define saturate(x) clamp(x, 0.0, 1.0)
#define lofi(i, m) (floor((i) / (m)) * (m))

// == hash / random ================================================================================
uvec3 seed;

// https://www.shadertoy.com/view/XlXcW4
vec3 hash3f(vec3 s) {
  uvec3 r = floatBitsToUint(s);
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  return vec3(r) / float(-1u);
}

vec3 uniformSphere(vec2 xi) {
  float phi = xi.x * TAU;
  float sinTheta = 1.0 - 2.0 * xi.y;
  float cosTheta = sqrt(1.0 - sinTheta * sinTheta);

  return vec3(
    cosTheta * cos(phi),
    cosTheta * sin(phi),
    sinTheta
  );
}

// == math utils ===================================================================================
mat2 rotate2D(float t) {
  return mat2(cos(t), -sin(t), sin(t), cos(t));
}

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 up = abs(z.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
  vec3 x = normalize(cross(up, z));
  return mat3(x, cross(z, x), z);
}

// == noise ========================================================================================
vec3 cyclic(vec3 p, float pers, float lacu) {
  vec4 sum = vec4(0);
  mat3 rot = orthBas(vec3(2, -3, 1));

  for (int i = 0; i ++ < 5;) {
    p *= rot;
    p += sin(p.zxy);
    sum += vec4(cross(cos(p), sin(p.yzx)), 1);
    sum /= pers;
    p *= lacu;
  }

  return sum.xyz / sum.w;
}

// == anim utils ===================================================================================
float ease(float t, float k) {
  float tt = fract(1.0 - t);
  return floor(t) + float(tt > 0.0) - (k + 1.0) * pow(tt, k) + k * pow(tt, k + 1.0);
}

// == 2d sdfs ======================================================================================
float sdcapsule2(vec2 p, vec2 tail) {
  float i_t = saturate(dot(p, tail) / dot(tail, tail));
  return length(p - i_t * tail);
}

// == text =========================================================================================
float sddomainchar(inout vec2 p, int code, float margin) {
  const ivec4 spaces[] = ivec4[](ivec4(3,5,8,8),ivec4(8,8,3,4),ivec4(4,8,8,3),ivec4(5,3,8,8),ivec4(8,8,8,8),ivec4(8,8,8,8),ivec4(8,3,3,8),ivec4(8,8,8,8),ivec4(8,8,8,8),ivec4(8,8,8,8),ivec4(3,8,8,8),ivec4(8,8,8,8),ivec4(8,8,8,8),ivec4(8,8,8,8),ivec4(8,8,4,8),ivec4(4,8,8,2));

  vec2 uv = saturate((p - vec2(4.0, 0.0)) / 16.0 + 0.5);
  float d = 100.0;
  if (abs(uv.x - 0.5) < 0.5 && abs(uv.y - 0.5) < 0.5) {
    uv = (uv + vec2(code % 8, code / 8)) / 8.0;
    d = texture(b1, uv).x;
  }

  p.x -= float(spaces[code / 4][code % 4]) + margin;

  return d;
}

// == primitive isects =============================================================================
vec4 isectBox(vec3 ro, vec3 rd, vec3 s) {
  vec3 xo = -ro / rd;
  vec3 xs = abs(s / rd);

  vec3 dfv = xo - xs;
  vec3 dbv = xo + xs;

  float df = max(dfv.x, max(dfv.y, dfv.z));
  float db = min(dbv.x, min(dbv.y, dbv.z));
  if (df < 0.0) { return vec4(FAR); }
  if (db < df) { return vec4(FAR); }

  vec3 n = -sign(rd) * step(vec3(df), dfv);
  return vec4(n, df);
}

vec4 isectIBox(vec3 ro, vec3 rd, vec3 s) {
  vec3 xo = -ro / rd;
  vec3 xs = abs(s / rd);

  vec3 dbv = xo + xs;

  float db = min(dbv.x, min(dbv.y, dbv.z));
  if (db < 0.0) { return vec4(FAR); }

  vec3 n = -sign(rd) * step(dbv, vec3(db));
  return vec4(n, db);
}

// == main =========================================================================================
void main() {
  float i_TENKAI_HELLO_RGB_DELAY = 32.0 + 0.5 * i_SWING;
  const float i_TENKAI_HELLO_HUGE_STUFF = 64.0;
  const float i_TENKAI_FLOOR_BEAT = 64.0;
  const float i_TENKAI_HELLO_LARGE_PILLAR = 96.0;
  const float i_TENKAI_RGB_DELAY_4FLOOR = 96.0;
  const float i_TENKAI_BREAK = 192.0;
  const float i_TENKAI_HELLO_RAINBOW_BAR = 224.0;
  const float i_TENKAI_HELLO_LASER = 224.0;
  const float i_TENKAI_FULLHOUSE = 224.0;
  const float i_TENKAI_TRANS = i_TENKAI_FULLHOUSE + 64.0;
  const float i_TENKAI_OUTRO = i_TENKAI_TRANS + 64.0;
  const float i_TENKAI_FADEOUT0 = i_TENKAI_OUTRO + 16.0;
  const float i_TENKAI_FADEOUT1 = i_TENKAI_FADEOUT0 + 16.0;

  outColor0 *= 0.0;

  vec2 uv = gl_FragCoord.xy / resolution.xy;

  float time = globalTime + 0.0 / i_B2T;
  vec3 seed = hash3f(vec3(uv, time));
  time += 0.003 * seed.z;
  float beats = time / i_B2T;
  float beatpulse = 0.4 + 0.6 * pow(0.5 - 0.5 * cos(TAU * ease(beats, 7.0)), 0.3) * (
    1.0 - 0.8 * smoothstep(0.0, 1.0, beats - i_TENKAI_BREAK - 1.0) * smoothstep(0.0, -0.5, beats - i_TENKAI_FULLHOUSE)
  );
  float beatpulse2 = exp(-5.0 * fract(beats));

  for (int i = 0; i ++ < i_SAMPLES;) {
    vec2 p = (uv - 0.5) + seed.xy / resolution.y;
    p.x *= resolution.x / resolution.y;

    vec3 colRem = vec3(0.4, 0.2, 1.0);

    float i_clen = 10.0;
    mat3 cb = orthBas(colRem);
    vec3 ro = i_clen * cb[2];
    vec3 rd = cb * normalize(vec3(p, -10.0));

    vec3 fp = ro + rd * (i_clen - 1.0);
    ro += cb * vec3(0.01 * tan(2.0 * (seed = hash3f(seed)).xy - 1.0).xy, 0.0);
    rd = normalize(fp - ro);
    ro += rd * i_clen * mix(0.5, 0.7, seed.z);

    float i_blur = exp(-0.2 * beats) + 0.04 * smoothstep(i_TENKAI_FADEOUT0, i_TENKAI_FADEOUT1, beats);
    ro += cb * vec3(i_blur * tan(2.0 * seed.xy - 1.0).xy, 0.0);

    ro.z -= 0.4 * time;

    colRem *= (1.0 - 0.5 * length(p)) / colRem;

    for (int i = 0; i ++ < i_REFLECTS;) {
      vec3 emissive = vec3(0.0);
      float roughness = 0.3;

      // floor
      vec4 isect2, isect = vec4(0.0, 1.0, 0.0, -ro.y / rd.y);
      if (isect.w < 0.0) {
        isect = vec4(FAR);
      }

      // floor greebles quadtree shit
      float grl = max(0.0, -(ro.y - i_GREEBLES_HEIGHT) / rd.y);

      for (int i = 0; i ++ < i_GREEBLES_TRAVERSAL;) {
        // if ray length is already further than isect, break
        if (isect.w < grl) {
          break;
        }

        // if already out of the greebles region, break
        vec3 gro = ro + rd * grl;
        if (gro.y * rd.y > 0.0 && abs(gro.y) > i_GREEBLES_HEIGHT) {
          break;
        }

        vec3 cell, dice, size = vec3(0.125, i_GREEBLES_HEIGHT, 0.125);
        for (int i = 0; i ++ < 4;) {
          if (i > 1) {
            if (dice.y < 0.4) {
              break;
            }
            size.xz /= 1.0 + vec2(step(0.6, dice.y), step(dice.y, 0.7));
          }

          cell = lofi(gro, 2.0 * size) + size;
          cell.y = 0.0;
          dice = hash3f(cell);
        }

        vec3 i_size = size - vec2(mix(1.0, 1.0 - beatpulse, step(i_TENKAI_FLOOR_BEAT, beats)) * (0.4 + 0.4 * sin(TAU * dice.z + time)) * i_GREEBLES_HEIGHT, i_GREEBLES_GAP).yxy;
        isect2 = isectBox(ro - cell, rd, i_size);
        if (isect2.w < isect.w) {
          isect = isect2;
          dice = hash3f(dice);
          emissive *= 0.0;
          roughness = exp(-1.0 - dice.y);
          break;
        }

        // forward to the next cell
        grl += isectIBox(gro - cell, rd, size).w + 0.01;
      }

      // plane array
      float mask = 0.0;
      float sidez = sign(rd.z);
      float planez = (floor(ro.z / i_PLANE_INTERVAL) + 0.5 * (1.0 + sidez)) * i_PLANE_INTERVAL;

      for (int i = 0; i ++ < i_PLANES_TRAVERSAL;) {
        isect2 = vec4(0.0, 0.0, -sidez, abs((ro.z - planez) / rd.z));

        // if the plane is already further than existing isect, break
        if (isect.w < isect2.w) {
          break;
        }

        vec3 rp = ro + rd * isect2.w;
        rp.y -= i_GREEBLES_HEIGHT;

        vec3 id = vec3(planez + vec3(1, 2, 3));
        vec3 dice = hash3f(id);

        float kind = floor(mod(planez / i_PLANE_INTERVAL, 8.0));
        if (kind == 4) {
          // rainbow bar
          if (abs(rp.y - 0.02) < 0.01 * ease(saturate(beats - i_TENKAI_HELLO_RAINBOW_BAR), 5.0)) {
            mask = 1.0;
            float i_phase = TAU * dice.z + rp.x;
            vec3 i_col = mix(
              1.0 + cos(i_phase + vec3(0, 2, 4)),
              vec3(smoothstep(2.0, 0.0, abs(rp.x)), 0.1, 1.0),
              ease(saturate(beats - i_TENKAI_TRANS), 3.0)
            );
            emissive += 10.0
              * exp(-40.0 * rp.y)
              * mix(1.0, sin(200.0 * rp.x), 0.2)
              * mix(1.0, sin(60.0 * (rp.x + beats)), 0.2)
              * mask
              * i_col
              * beatpulse;
          }

          // warning
          rp.y -= 0.05;
          float warningheight = 0.025 * ease(saturate(beats - i_TENKAI_BREAK), 5.0) * smoothstep(0.0, -1.0, beats - i_TENKAI_FULLHOUSE);
          if (abs(rp.y) < warningheight && i_TENKAI_BREAK <= beats && beats < i_TENKAI_FULLHOUSE) {
            mask = 1.0;

            rp.x = mod(rp.x + 0.1 * time, 1.0) - 0.5;
            float blind = step(fract(20.0 * (rp.x + rp.y + 0.1 * time)), 0.5) * step(0.3, abs(rp.x)) * step(abs(rp.y), warningheight - 0.008);

            float shape = max(
              step(texture(b1, saturate(rp.xy / 24.0 / warningheight + 0.5)).y, 1.0),
              blind
            );

            emissive += mix(
              mix(
                vec3(1.0, 0.04, 0.04),
                vec3(1.0, 0.5, 0.04),
                mod(floor(beats), 2.0)
              ),
              mix(
                vec3(1.0),
                vec3(0.0),
                mod(floor(beats), 2.0)
              ),
              shape
            );

          }
        } else if (kind == 0) {
          // large pillar
          float i_ratio = ease(saturate(beats - i_TENKAI_HELLO_LARGE_PILLAR), 3.0);
          rp.x = abs(abs(rp.x) - 0.5) / 0.05 / i_ratio;
          if (rp.x < 1.0) {
            mask = 1.0;
            vec3 i_col = exp(-rp.y) * mix(
              vec3(4.0, 6.0, 8.0),
              vec3(9.0 * exp(-4.0 * rp.y), 0.5, 8.0),
              ease(saturate(beats - i_TENKAI_TRANS), 3.0)
            ) * beatpulse * cos(0.9 * rp.x);
            emissive += i_col;
          }
        } else if (kind == 2) {
          // rave laser
          rp.y += 0.01;
          float t = dice.y + floor(beats);
          float d = min(
            max(abs(mod((rp.xy * rotate2D(t)).x, 0.04) - 0.02), 0.0),
            max(abs(mod((rp.xy * rotate2D(-t)).x, 0.04) - 0.02), 0.0)
          );
          vec3 i_col = mix(
            vec3(0.1, 10.0, 2.0),
            vec3(10.0, 0.1, 0.1),
            ease(saturate(beats - i_TENKAI_TRANS), 3.0)
          );
          emissive += step(i_TENKAI_HELLO_LASER, beats) * smoothstep(2.0, 0.0, abs(rp.x)) * exp(-4.0 * abs(rp.y)) * beatpulse2 * step(d, 0.001) * i_col;
        } else if (kind == 6) {
          if (i_TENKAI_HELLO_HUGE_STUFF <= beats && beats < i_TENKAI_BREAK || i_TENKAI_FULLHOUSE <= beats && beats < i_TENKAI_OUTRO) {
            // huge stuff
            dice = hash3f(dice + floor(beats));
            rp.x += floor(9.0 * dice.y - 4.0) * 0.25;

            if (dice.x < 0.25) {
              // pillars
              mask = step(abs(rp.x), 0.125) * step(abs(fract(64.0 * rp.x) - 0.5), 0.05);
            } else if (dice.x < 0.5) {
              // x
              rp.y -= 0.25;
              float i_range = max(abs(rp.x) - 0.25, abs(rp.y) - 0.25);
              mask = max(
                step(abs(rp.x + rp.y), 0.002),
                step(abs(rp.x - rp.y), 0.002)
              ) * step(i_range, 0.0);
            } else if (dice.x < 0.75) {
              // dashed box
              dice.yz = exp(-3.0 * dice.yz);
              rp.y -= dice.z;
              float d = max(abs(rp.x) - dice.y, abs(rp.y) - dice.z);
              float shape = step(abs(d), 0.001) * step(0.5, fract(dot(rp, vec3(32.0)) + time));
              mask = shape;
            } else {
              // huge circle
              rp.y -= 0.5;
              mask = step(abs(length(rp.xy) - 0.5), 0.001);
            }

            emissive += 10.0 * beatpulse2 * mask;
            mask = 0.0;
          }
        } else if (abs(rp.x) < 1.0 && i_TENKAI_HELLO_RGB_DELAY <= beats) {
          // rgb delay shit
          float size = 0.25;
          dice = hash3f(vec3(floor(rp.xy / size), dice.z));
          size /= 1.0 + step(0.3, dice.z);
          dice = hash3f(vec3(floor(rp.xy / size), dice.z));
          size /= 1.0 + step(0.5, dice.z);
          dice = hash3f(vec3(floor(rp.xy / size), dice.z));
          vec2 cp = rp.xy / size;

          if (abs(cp.y - 0.5) < 0.5) {
            cp = (fract(cp.xy) - 0.5) * size / (size - 0.01);

            if (abs(cp.x) < 0.5 && abs(cp.y) < 0.5) {
              float off = (seed = hash3f(seed)).y;
              float beatsoff = beats - 0.2 * off + 0.1;
              vec3 col = 4.0 * 3.0 * (0.5 - 0.5 * cos(TAU * saturate(1.5 * off - vec3(0.0, 0.25, 0.5)))) * (1.0 + sin(400.0 * rp.y + 100.0 * beatsoff));
              float timegroup = floor(4.0 * dice.x);

              if (beatsoff < i_TENKAI_RGB_DELAY_4FLOOR) {
                // b2sSwing
                float st = 4.0 * beatsoff;
                st = 2.0 * floor(st / 2.0) + step(i_SWING, fract(0.5 * st));

                st = clamp(st, 4.0 * i_TENKAI_HELLO_RGB_DELAY + 10.0, 4.0 * i_TENKAI_RGB_DELAY_4FLOOR - 16.0);
                st += floor(st / 32.0);
                st -= 1.0 + 3.0 * timegroup;
                st = lofi(st, 12.0);
                st += 1.0 + 3.0 * timegroup;
                st -= floor(st / 32.0);

                float i_bst = 0.5 * (floor(st / 2.0) + i_SWING * mod(st, 2.0));
                float t = beatsoff - i_bst;

                col *= vec3(1.0, 0.04, 0.1) * step(0.0, t) * (exp(-4.0 * t) + exp(-40.0 * t));
              } else if (beatsoff < i_TENKAI_BREAK) {
                float b = beatsoff;

                b = clamp(b, i_TENKAI_RGB_DELAY_4FLOOR + 3.0, i_TENKAI_FULLHOUSE);
                b -= timegroup;
                b = lofi(b, 4.0);
                b += timegroup;

                float t = beatsoff - b;

                col *= step(0.0, t) * (exp(-2.0 * t) + exp(-20.0 * t)) * (0.5 + 0.5 * cos(PI * 5.0 * saturate(2.0 * (beatsoff - i_TENKAI_BREAK + 0.5))));
              } else {
                float thr = sqrt(fract(dice.x * 999.0));

                col *= smoothstep(0.0, 4.0, beatsoff - i_TENKAI_BREAK - 32.0 * thr) * mix(vec3(1.0), vec3(1.0, 0.05, 0.12), ease(saturate(beats - i_TENKAI_TRANS), 3.0));
              }

              col = max(col, 0.0);

              float phase = (
                1.0
                + max(beatsoff - timegroup - i_TENKAI_BREAK, 0.0) / 4.0
                + max(beatsoff - timegroup - i_TENKAI_FULLHOUSE, 0.0) / 4.0
              );

              float ephase = ease(phase, 6.0);
              float ephase0 = min(mod(ephase, 2.0), 1.0);
              float ephase1 = max(mod(ephase, 2.0) - 1.0, 0.0);

              dice.z *= 24;

              if (dice.z < 1) {
                // ,',
                cp *= rotate2D(3.0 * PI * ephase);
                float theta = lofi(atan(cp.x, cp.y), TAU / 3.0) + PI / 3.0;
                cp = (cp * rotate2D(theta) - vec2(0.0, 0.3));
                float shape = step(length(cp), 0.1);
                emissive += col * shape;
              } else if (dice.z < 2) {
                // circle
                emissive += col * step(0.5 * ephase0 - 0.2, length(cp)) * step(length(cp), 0.5 * ephase0) * step(1.1 * ephase1, fract(atan(cp.y, cp.x) / TAU - ephase1 - 2.0 * TAU * dice.y));
              } else if (dice.z < 3) {
                // slide
                cp.x *= sign(dice.y - 0.5);
                cp *= rotate2D(PI / 4.0);
                cp.x += 2.0 * sign(cp.y) * (1.0 - ephase0);
                cp = abs(cp);
                float shape = step(0.03 + ephase1, cp.y) * step(max(cp.x, cp.y), 0.65) * step(cp.x + cp.y, 1.0 / SQRT2);
                emissive += col * shape;
              } else if (dice.z < 4) {
                // dot matrix
                float shape = step(abs(cp.y), 0.5) * step(abs(cp.x), 0.5);
                cp *= 4.0;
                shape *= step(length(fract(cp) - 0.5), 0.4);
                cp = floor(cp);
                float i_rand = floor(12.0 * min(fract(phase), 0.5));
                emissive += col * shape * step(
                  hash3f(vec3(cp, dice.y + i_rand)).x,
                  0.3 - 0.3 * cos(PI * ephase)
                );
              } else if (dice.z < 5) {
                // target
                cp = abs(cp);
                float i_shape = max(
                  step(abs(max(cp.x, cp.y) - 0.48), 0.02) * step(0.8 - 0.6 * ephase0, min(cp.x, cp.y)),
                  step(max(cp.x, cp.y), 0.15 * ephase0) * step(abs(min(cp.x, cp.y)), 0.02)
                ) * step(fract(3.0 * max(ephase1, 0.5)), 0.5);
                emissive += col * i_shape;
              } else if (dice.z < 6) {
                // hex
                cp *= rotate2D(TAU * lofi(dice.y - ephase, 1.0 / 6.0));
                float cell = floor(atan(cp.x, cp.y) / TAU * 6.0 + 0.5);
                cp *= rotate2D(cell / 6.0 * TAU);
                float i_shape = (
                  step(0.02, dot(abs(cp), vec2(-SQRT3_OVER_TWO, 0.5)))
                  * step(0.24, cp.y)
                  * step(cp.y, 0.44)
                ) * step(mod(cell, 3.0), 1.0 - 1.1 * cos(PI * ephase));
                emissive += col * i_shape;
              } else if (dice.z < 7) {
                // 0b5vr (hide in the first half)
                cp = floor(8.0 * cp + 0.5);
                float i_lcp = length(cp);
                float i_rand = floor(16.0 * phase);
                float i_obsvr = step(i_lcp, 0.5) + step(1.5, i_lcp) * step(i_lcp, 2.5);
                float i_shape = step(
                  hash3f(vec3(cp, dice.y + i_rand)).x,
                  0.5 + 0.5 * cos(PI * ephase)
                ) * i_obsvr;
                emissive += col * i_shape;
              } else if (dice.z < 8) {
                // char
                float i_rand = floor(30.0 * min(fract(phase), 0.2)) + floor(phase);
                int i_char = int(64.0 * hash3f(dice + i_rand).x);
                cp = 12.0 * cp + vec2(4.0, 0.0);
                float i_d = sddomainchar(cp, i_char, 0.0);
                emissive += col * step(i_d, 0.5);
              } else if (dice.z < 12) {
                // arrow
                cp /= 0.001 + ephase0;

                float blink = floor(min(8.0 * ephase1, 3.0));

                float dout, din = 1.0;

                if (dice.z < 9) {
                  // arrow
                  vec2 cpt = vec2(
                    abs(cp.x),
                    0.8 - fract(cp.y + 0.5 - 2.0 * ephase0)
                  );

                  din = min(
                    sdcapsule2(cpt, vec2(0.0, 0.6)),
                    sdcapsule2(cpt, vec2(0.3, 0.3))
                  ) - 0.07;

                  cpt = cp;
                  cpt -= clamp(cpt, -0.4, 0.4);
                  dout = length(cpt) - 0.05;
                } else if (dice.z < 10) {
                  // error
                  dout = length(cp) - 0.45;

                  cp *= rotate2D(PI * ephase0 + PI / 4.0);
                  cp = abs(cp);

                  din = max(
                    max(cp.x, cp.y) - 0.25,
                    min(cp.x, cp.y) - 0.07
                  );
                } else if (dice.z < 11) {
                  // warning
                  cp.x = abs(cp.x);
                  din = max(
                    cp.x - 0.07,
                    min(
                      abs(cp.y) - 0.15,
                      abs(cp.y + 0.27) - 0.05
                    )
                  ) + step(fract(3.9 * ephase0), 0.5);

                  dout = mix(
                    min(
                      sdcapsule2(cp - vec2(0.0, 0.35), vec2(0.4, -0.7)),
                      sdcapsule2(cp + vec2(0.0, 0.35), vec2(0.4, 0.0))
                    ),
                    0.0,
                    step(dot(cp, vec2(0.7, 0.4)), 0.11) * step(-0.4, cp.y) // cringe
                  ) - 0.05;
                } else {
                  // power
                  dout = 0.3 * ephase0 * ephase0;
                  dout = sdcapsule2(cp - vec2(0.0, 0.1), vec2(0.0, dout)) - 0.07;
                  float i_ring = max(
                    abs(length(cp) - 0.4) - 0.07,
                    -dout + 0.07
                  );
                  dout = min(dout, i_ring);
                }

                float i_shape = mix(
                  mix(
                    step(max(dout, -din), 0.0),
                    step(abs(max(dout, -din)), 0.01),
                    saturate(blink)
                  ),
                  mix(
                    step(din, 0.0),
                    0.0,
                    saturate(blink - 2.0)
                  ),
                  saturate(blink - 1.0)
                );
                emissive += col * i_shape;
              }
            }
          }
        }

        // if the mask test misses, traverse the next plane
        if (mask == 0.0) {
          planez += i_PLANE_INTERVAL * sidez;
          continue;
        }

        // hit!
        isect = isect2;
        roughness = 0.0;
        break;
      }

      // emissive
      outColor0.xyz += colRem * emissive;

      // if mask is set, break
      if (mask > 0.0) {
        break;
      }

      // the ray missed all of the above, you suck
      if (isect.w >= FAR) {
        float i_intro = 0.5 * smoothstep(0.0, 32.0, beats) * (0.01 + smoothstep(32.0, 31.5, beats));
        outColor0.xyz += colRem * i_intro;
        break;
      }

      // now we have a hit

      // set materials
      ro += isect.w * rd + isect.xyz * 0.001;
      float sqRoughness = roughness * roughness;
      float sqSqRoughness = sqRoughness * sqRoughness;
      float halfSqRoughness = 0.5 * sqRoughness;

      // shading
      {
        float NdotV = dot(isect.xyz, -rd);
        float Fn = mix(0.04, 1.0, pow(1.0 - NdotV, 5.0));
        float spec = 1.0;

        // sample ggx or lambert
        seed.y = sqrt((1.0 - seed.y) / (1.0 - (1.0 - sqSqRoughness) * seed.y));
        vec3 i_H = orthBas(isect.xyz) * vec3(
          sqrt(1.0 - seed.y * seed.y) * sin(TAU * seed.z + vec2(0.0, TAU / 4.0)),
          seed.y
        );

        // specular
        vec3 wo = reflect(rd, i_H);
        if (dot(wo, isect.xyz) < 0.0) {
          break;
        }

        // vector math
        float NdotL = dot(isect.xyz, wo);
        float i_VdotH = dot(-rd, i_H);
        float i_NdotH = dot(isect.xyz, i_H);

        // fresnel
        vec3 i_baseColor = vec3(0.3);
        vec3 i_F0 = i_baseColor;
        vec3 i_Fh = mix(i_F0, vec3(1.0), pow(1.0 - i_VdotH, 5.0));

        // brdf
        // colRem *= Fh / Fn * G * VdotH / ( NdotH * NdotV );
        colRem *= saturate(
          i_Fh
            / (NdotV * (1.0 - halfSqRoughness) + halfSqRoughness) // G1V / NdotV
            * NdotL / (NdotL * (1.0 - halfSqRoughness) + halfSqRoughness) // G1L
            * i_VdotH / i_NdotH
        );

        // prepare the rd for the next ray
        rd = wo;
      }

      if (dot(colRem, colRem) < 0.01) {
        break;
      }
    }

    // title
    if (beats < i_TENKAI_HELLO_RGB_DELAY) {
      float phase = (float(i - 1) + seed.x) / i_SAMPLES_F;
      float diffuse = phase * phase * phase * phase;
      p += (exp(-0.08 * beats) * diffuse + 0.5 * exp(-0.4 * beats) * phase) * cyclic(vec3(4.0 * p, 0.2 * time) + 5.0, 0.5, 1.4).xy;

      float d = 100.0;

      d = texture(b1, saturate(0.7 * p + 0.5)).y;

      // render
      float shape = smoothstep(2.0 * diffuse, 0.0, d - 0.2);
      vec3 i_col = 3.0 * (0.5 - 0.5 * cos(TAU * saturate(1.5 * phase - vec3(0.0, 0.25, 0.5))));
      outColor0.xyz += shape * i_col * smoothstep(-1.0, -4.0, beats - i_TENKAI_HELLO_RGB_DELAY);
    }
  }

  outColor0.xyz = mix(
    smoothstep(
      vec3(-0.0, -0.1, -0.2),
      vec3(1.0, 1.1, 1.2),
      sqrt(outColor0.xyz / i_SAMPLES_F)
    ),
    max(texture(b0, uv), 0.0).xyz,
    0.5
  ) * smoothstep(0.0, 4.0, beats) * smoothstep(i_TENKAI_FADEOUT1, i_TENKAI_FADEOUT0, beats);

  // -- buffer 1 green - texts ---------------------------------------------------------------------
  // title, "planefiller"
  if (beats < i_TENKAI_HELLO_RGB_DELAY) {
    float d = 100.0;

    vec2 p = (uv - 0.5) * 160.0 + vec2(56.0, 0.0);

    float i_margin = 3.0;
    d = min(d, sddomainchar(p, 47, i_margin));
    d = min(d, sddomainchar(p, 43, i_margin));
    d = min(d, sddomainchar(p, 32, i_margin));
    d = min(d, sddomainchar(p, 45, i_margin));
    d = min(d, sddomainchar(p, 36, i_margin));
    d = min(d, sddomainchar(p, 37, i_margin));
    d = min(d, sddomainchar(p, 40, i_margin));
    d = min(d, sddomainchar(p, 43, i_margin));
    d = min(d, sddomainchar(p, 43, i_margin));
    d = min(d, sddomainchar(p, 36, i_margin));
    d = min(d, sddomainchar(p, 49, i_margin));

    // render
    outColor1.y = d;
  }

  // "warning"
  if (i_TENKAI_BREAK <= beats && beats < i_TENKAI_FULLHOUSE - 4.0) {
    const int codes[] = int[](
      0,
      54, 32, 49, 45, 40, 45, 38, // warning
      63, 12, 63, // -
      35, 49, 46, 47, 63, // drop
      40, 45, 34, 46, 44, 40, 45, 38
    );

    float d = 100.0;

    vec2 p = (uv - 0.5) * 280.0 + vec2(118.0, 0.0);

    for (int i = 0; i ++ < 23;) {
      float phase = saturate((beats - i_TENKAI_BREAK - 0.5 - 0.05 * float(i)) / 0.25);
      if (0.0 < phase) {
        int i_offset = 3 * (int(16.0 * phase) - 16);
        int code = (codes[i] - i_offset) % 64;
        d = min(d, sddomainchar(p, code, 4.0));
      }
    }

    // render
    outColor1.y = d;
  }

  // countdown
  if (i_TENKAI_FULLHOUSE - 4.0 <= beats && beats < i_TENKAI_FULLHOUSE) {
    vec2 p = (uv - 0.5) * 280.0 + vec2(4.0, 0.0);

    float d = sddomainchar(p, 18 - int(beats) % 4, 4.0);

    // render
    outColor1.y = d;
  }

  // -- buffer 1 red - chars -----------------------------------------------------------------------
  outColor1.x = texture(b1, uv).x;
  if (outColor1.x == 0.0) {
    const ivec4 vertices[] = ivec4[](ivec4(0x10111216,0x35361516,0x61016505,0x50561016),ivec4(0x05165665,0x62531304,0x01105061,0x26063036),ivec4(0x42060424,0x42406062,0x00016566,0x41425363),ivec4(0x02011030,0x45443313,0x04051636,0x41423313),ivec4(0x15166050,0x30212536,0x00111506,0x03653236),ivec4(0x31356305,0x00116303,0x10114303,0x00016566),ivec4(0x65615010,0x01051656,0x16016510,0x10303526),ivec4(0x56160550,0x13536465,0x05600002,0x64655616),ivec4(0x62533353,0x01105061,0x63130406,0x66066066),ivec4(0x63540406,0x01105061,0x01051656,0x62615010),ivec4(0x66060353,0x56101165,0x13040516,0x50616253),ivec4(0x13020110,0x56656453,0x05041363,0x61655616),ivec4(0x14011050,0x14101113,0x65001113,0x64046103),ivec4(0x63056101,0x56160501,0x33536465,0x36303132),ivec4(0x63524233,0x05165665,0x00501001,0x63462603),ivec4(0x00620260,0x64655606,0x62530353,0x65005061),ivec4(0x01051656,0x06615010,0x65560600,0x06005061),ivec4(0x06630366,0x66066000,0x00065303,0x05165665),ivec4(0x61501001,0x00064363,0x60666303,0x61661016),ivec4(0x02011050,0x65660006,0x61430343,0x60000660),ivec4(0x32330600,0x00606633,0x10666006,0x56656150),ivec4(0x10010516,0x65560600,0x10035364,0x56656150),ivec4(0x10010516,0x06006142,0x53646556,0x60625303),ivec4(0x05165665,0x62531304,0x01105061,0x30366606),ivec4(0x50100106,0x03066661,0x66634020,0x34330006),ivec4(0x06666033,0x66606105,0x06000165,0x65663305),ivec4(0x66063033,0x60000165,0x30202636,0x60610506),ivec4(0x00101606,0x00543614,0x00000060,0));
    const ivec4 segments[] = ivec4[](ivec4(0,2,4,6),ivec4(8,10,12,14),ivec4(16,28,30,35),ivec4(40,44,66,68),ivec4(72,76,78,80),ivec4(82,84,86,88),ivec4(90,92,96,105),ivec4(107,111,113,123),ivec4(130,136,140,142),ivec4(144,152,162,167),ivec4(184,195,197,199),ivec4(201,203,206,208),ivec4(210,213,221,223),ivec4(235,241,243,250),ivec4(255,263,265,271),ivec4(273,275,278,280),ivec4(282,284,294,296),ivec4(298,300,302,308),ivec4(310,314,317,320),ivec4(324,327,331,340),ivec4(347,356,358,365),ivec4(368,380,382,384),ivec4(390,396,400,403),ivec4(407,411,414,418),ivec4(424,428,432,436),ivec4(439,441,0,0));
    const ivec4 chars[] = ivec4[](ivec4(0,2,4,8),ivec4(10,13,14,15),ivec4(16,17,20,22),ivec4(23,24,25,26),ivec4(28,30,31,33),ivec4(35,37,38,39),ivec4(40,41,43,45),ivec4(46,48,49,51),ivec4(52,54,56,57),ivec4(59,62,65,66),ivec4(69,70,71,74),ivec4(75,77,78,79),ivec4(80,82,84,85),ivec4(87,88,89,91),ivec4(93,95,96,97),ivec4(98,99,100,101));

    const ivec4 spaces[] = ivec4[](ivec4(3,5,8,8),ivec4(8,8,3,4),ivec4(4,8,8,3),ivec4(5,3,8,8),ivec4(8,8,8,8),ivec4(8,8,8,8),ivec4(8,3,3,8),ivec4(8,8,8,8),ivec4(8,8,8,8),ivec4(8,8,8,8),ivec4(3,8,8,8),ivec4(8,8,8,8),ivec4(8,8,8,8),ivec4(8,8,8,8),ivec4(8,8,4,8),ivec4(4,8,8,8));

    int code = int(dot(floor(8 * uv), vec2(1, 8)));
    vec2 p = (fract(8 * uv) - 0.5) * 16.0;

    float d = 100.0;

    int seg0 = chars[code / 4][code % 4];
    code ++;
    int seg1 = chars[code / 4][code % 4];

    for (int i = seg0; i ++ < seg1;) {
      int j = i - 1;
      int vert0 = segments[j / 4][j % 4];
      j ++;
      int vert1 = segments[j / 4][j % 4] - 1;

      for (int i = vert0; i ++ < vert1;) {
        int j = i - 1;
        int i_iv0 = (vertices[j / 16][j / 4 % 4] >> ((j % 4) * 8));
        vec2 v0 = vec2((i_iv0 / ivec2(16, 1)) & 15);
        j ++;
        int i_iv1 = (vertices[j / 16][j / 4 % 4] >> ((j % 4) * 8));
        vec2 v1 = vec2((i_iv1 / ivec2(16, 1)) & 15);

        v0 = 1.5 * v0 + vec2(-0.5, 0.5) * (step(2.0, v0) + step(5.0, v0)) - vec2(4.0, 5.0);
        v1 = 1.5 * v1 + vec2(-0.5, 0.5) * (step(2.0, v1) + step(5.0, v1)) - vec2(4.0, 5.0);
        d = min(d, sdcapsule2(p - v0, v1 - v0));
      }
    }

    outColor1.x = d;
  }
}
