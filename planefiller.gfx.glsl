#version 430

layout(binding = 0) uniform sampler2D backBuffer0;
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

out vec4 outColor;

// == constants ====================================================================================
const float FAR = 44.0;

const float PI = acos(-1.0);
const float TAU = PI + PI;
const float SQRT2 = sqrt(2.0);
const float SQRT3 = sqrt(3.0);
const float SQRT3_OVER_TWO = SQRT3 / 2.0;

const float i_B2T = 0.43;
const float i_SWING = 0.62;
const int i_SAMPLES = 16;
const float i_SAMPLES_F = 16.0;
const int i_REFLECTS = 3;

const float i_GAP = 0.01;
const float i_GREEBLES_HEIGHT = 0.02;

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
float sddomainsegment(vec2 p, vec2 a, vec2 b) {
  a = 1.5 * a + vec2(-0.5, 1.5) * (step(2.0, a) + step(5.0, a)) - vec2(0.0, 6.0);
  b = 1.5 * b + vec2(-0.5, 1.5) * (step(2.0, b) + step(5.0, b)) - vec2(0.0, 6.0);
  return sdcapsule2(p - a, b - a);
}

float sddomainchar(vec2 p, int code) {
  const ivec4 vertices[] = ivec4[](ivec4(0x016121110,0x016153635,0x005650161,0x016105650),ivec4(0x065561605,0x004135362,0x061501001,0x036300626),ivec4(0x024040642,0x062604042,0x066650100,0x063534241),ivec4(0x030100102,0x013334445,0x036160504,0x013334241),ivec4(0x050601615,0x036252130,0x006151100,0x036326503),ivec4(0x005633531,0x003631100,0x003431110,0x066650100),ivec4(0x010506165,0x056160501,0x010650116,0x026353010),ivec4(0x050051656,0x065645313,0x002006005,0x016566564),ivec4(0x053335362,0x061501001,0x006041363,0x066600666),ivec4(0x006045463,0x061501001,0x056160501,0x010506162),ivec4(0x053030666,0x065111056,0x016050413,0x053626150),ivec4(0x010010213,0x053646556,0x063130405,0x016566561),ivec4(0x050100114,0x013111014,0x013110065,0x003610464),ivec4(0x001610563,0x001051656,0x065645333,0x032313036),ivec4(0x033425263,0x065561605,0x001105000,0x003264663),ivec4(0x060026200,0x006566564,0x053035362,0x061500065),ivec4(0x056160501,0x010506106,0x000065665,0x061500006),ivec4(0x066036306,0x000600666,0x003530600,0x065561605),ivec4(0x001105061,0x063430600,0x003636660,0x016106661),ivec4(0x050100102,0x006006665,0x043034361,0x060060060),ivec4(0x000063332,0x033666000,0x006606610,0x050616556),ivec4(0x016050110,0x000065665,0x064530310,0x050616556),ivec4(0x016050110,0x042610006,0x056656453,0x003536260),ivec4(0x065561605,0x004135362,0x061501001,0x006663630),ivec4(0x006011050,0x061660603,0x020406366,0x006003334),ivec4(0x033606606,0x005616066,0x065010006,0x005336665),ivec4(0x033300666,0x065010060,0x036262030,0x006056160),ivec4(0x006161000,0x014365400,0x060000000,0));
  const ivec4 segments[] = ivec4[](ivec4(0,2,4,6),ivec4(8,10,12,14),ivec4(16,28,30,35),ivec4(40,44,66,68),ivec4(72,76,78,80),ivec4(82,84,86,88),ivec4(90,92,96,105),ivec4(107,111,113,123),ivec4(130,136,140,142),ivec4(144,152,162,167),ivec4(184,195,197,199),ivec4(201,203,206,208),ivec4(210,213,221,223),ivec4(235,241,243,250),ivec4(255,263,265,271),ivec4(273,275,278,280),ivec4(282,284,294,296),ivec4(298,300,302,308),ivec4(310,314,317,320),ivec4(324,327,331,340),ivec4(347,356,358,365),ivec4(368,380,382,384),ivec4(390,396,400,403),ivec4(407,411,414,418),ivec4(424,428,432,436),ivec4(439,441,0,0));
  const ivec4 chars[] = ivec4[](ivec4(0,2,4,8),ivec4(10,13,14,15),ivec4(16,17,20,22),ivec4(23,24,25,26),ivec4(28,30,31,33),ivec4(35,37,38,39),ivec4(40,41,43,45),ivec4(46,48,49,51),ivec4(52,54,56,57),ivec4(59,62,65,66),ivec4(69,70,71,74),ivec4(75,77,78,79),ivec4(80,82,84,85),ivec4(87,88,89,91),ivec4(93,95,96,97),ivec4(98,99,100,101));

  float d = 100.0;

  int seg0 = chars[code / 4][code % 4];
  code ++;
  int seg1 = chars[code / 4][code % 4];

  for (int i = seg0; i < seg1;) {
    int vert0 = segments[i / 4][i % 4];
    i ++;
    int vert1 = segments[i / 4][i % 4] - 1;

    for (int j = vert0; j < vert1;) {
      int v0 = (vertices[j / 16][j / 4 % 4] >> (8 * (3 - j % 4))) & 255;
      j ++;
      int v1 = (vertices[j / 16][j / 4 % 4] >> (8 * (3 - j % 4))) & 255;

      d = min(d, sddomainsegment(
        p,
        vec2(v0 >> 4, v0 & 15),
        vec2(v1 >> 4, v1 & 15)
      ));
    }
  }

  return d;
}

float sddomainspace(int code) {
  const ivec4 spaces[] = ivec4[](
    ivec4(3,5,8,8),ivec4(8,8,3,4),ivec4(4,8,8,3),ivec4(5,3,8,8),
    ivec4(8,8,8,8),ivec4(8,8,8,8),ivec4(8,3,3,8),ivec4(8,8,8,8),
    ivec4(8,8,8,8),ivec4(8,8,8,8),ivec4(3,8,8,8),ivec4(8,8,8,8),
    ivec4(8,8,8,8),ivec4(8,8,8,8),ivec4(8,8,4,8),ivec4(4,8,8,8)
  );
  return float(spaces[code / 4][code % 4]);
}

float sddomaincharspace(inout vec2 p, int code, float padding) {
  float d = sddomainchar(p, code);
  p.x -= sddomainspace(code) + padding;
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
  const float i_TENKAI_HELLO_GRID = 96.0;
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

  outColor *= 0.0;

  vec2 uv = gl_FragCoord.xy / resolution.xy;

  float time = globalTime + 0.0 / i_B2T;
  vec3 seed = hash3f(vec3(uv, time));
  time += 0.01 * seed.z;
  float beats = time / i_B2T;
  float beatpulse = 0.2 + 0.8 * pow(0.5 - 0.5 * cos(TAU * ease(beats, 7.0)), 0.3);
  float beatpulse2 = exp(-5.0 * fract(beats));

  for (int i = 0; i ++ < i_SAMPLES;) {
    vec2 p = (uv - 0.5) + seed.xy / resolution.y;
    p.x *= resolution.x / resolution.y;

    vec3 colRem = vec3(0.4, 0.2, 1.0);

    mat3 cb = orthBas(colRem);
    vec3 ro = 10.0 * cb[2];
    vec3 rd = cb * normalize(vec3(p, -10.0));

    vec3 fp = ro + rd * 9.0;
    ro += cb * vec3(0.01 * tan(2.0 * (seed = hash3f(seed)).xy - 1.0).xy, 0.0);
    rd = normalize(fp - ro);
    ro += rd * mix(5.0, 6.0, seed.z);

    float i_blur = exp(-0.2 * beats) + 0.04 * smoothstep(i_TENKAI_FADEOUT0, i_TENKAI_FADEOUT1, beats);
    ro += cb * vec3(i_blur * tan(2.0 * seed.xy - 1.0).xy, 0.0);

    ro.z -= 0.4 * time;

    colRem *= (1.0 - 0.5 * length(p)) / colRem;

    const float i_PLANE_INTERVAL = 0.25;

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

      for (int i = 0; i ++ < 8;) {
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

        vec3 i_size = size - vec2(mix(1.0, 1.0 - beatpulse, step(i_TENKAI_FLOOR_BEAT, beats)) * (0.4 + 0.4 * sin(TAU * dice.z + time)) * i_GREEBLES_HEIGHT, i_GAP).yxy;
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

      for (int i = 0; i ++ < 32;) {
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
        if (kind == 0) {
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
              * mix(1.0, sin(40.0 * (rp.x + beats)), 0.2)
              * mask
              * i_col
              * beatpulse;
          }

          // warning
          rp.y -= 0.05;
          float warningwidth = 0.025 * ease(saturate(beats - i_TENKAI_BREAK), 5.0) * smoothstep(0.0, -1.0, beats - i_TENKAI_FULLHOUSE);
          if (abs(rp.y) < warningwidth) {
            const int codes[] = int[](0, 54, 32, 49, 45, 40, 45, 38);

            mask = 1.0;

            rp.x = mod(rp.x + 0.1 * time, 0.5) - 0.25;
            float blind = step(fract(20.0 * (rp.x + rp.y + 0.1 * time)), 0.5) * step(0.12, abs(rp.x)) * step(abs(rp.y), warningwidth - 0.008);

            rp.xy *= 12.0 / warningwidth;
            for (int i = 0; i ++ < 7;) {
              float phase = saturate((beats - i_TENKAI_BREAK - 0.5 - 0.1 * float(i)) / 0.25);
              if (0.0 < phase) {
                int i_offset = int(16.0 * phase) - 16;
                int code = (codes[i] - i_offset) % 64;
                rp.x += 0.5 * sddomainspace(code) + 2.0;
              }
            }
            rp.x -= 2.0;

            float d = 100.0;
            for (int i = 0; i ++ < 7;) {
              float phase = saturate((beats - i_TENKAI_BREAK - 0.5 - 0.1 * float(i)) / 0.25);
              if (0.0 < phase) {
                int i_offset = int(16.0 * phase) - 16;
                int code = (codes[i] - i_offset) % 64;
                d = min(d, sddomaincharspace(rp.xy, code, 4.0) - 1.0);
              }
            }
            float shape = max(
              step(d, 0.0),
              blind
            );

            emissive += mix(
              vec3(1.0, 0.04, 0.04),
              vec3(1.0),
              shape
            );

          }
        } else if (kind == 4) {
          // large pillar
          float i_ratio = ease(saturate(beats - i_TENKAI_HELLO_LARGE_PILLAR), 3.0);
          mask = step(abs(abs(rp.x) - 0.5), 0.05 * i_ratio);
          vec3 i_col = exp(-rp.y) * mix(
            vec3(4.0, 6.0, 8.0),
            vec3(9.0 * exp(-4.0 * rp.y), 0.5, 8.0),
            ease(saturate(beats - i_TENKAI_TRANS), 3.0)
          ) * mix(
            beatpulse,
            0.1,
            smoothstep(0.0, 1.0, beats - i_TENKAI_BREAK - 1.0) * smoothstep(0.0, -0.5, beats - i_TENKAI_FULLHOUSE)
          );
          emissive += i_col * mask;
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
          emissive += step(i_TENKAI_HELLO_LASER, beats) * smoothstep(2.0, 0.0, abs(rp.x)) * exp(-4.0 * rp.y) * beatpulse2 * step(d, 0.001) * i_col;
        } else if (kind == 6) {
          if (i_TENKAI_HELLO_HUGE_STUFF <= beats && beats < i_TENKAI_BREAK || i_TENKAI_FULLHOUSE <= beats && beats < i_TENKAI_OUTRO) {
            // huge stuff
            dice = hash3f(dice + floor(beats));
            rp.x += floor(17.0 * dice.y - 8.0) * 0.25;

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
              vec3 col = 4.0 * 3.0 * (0.5 - 0.5 * cos(TAU * saturate(1.5 * off - vec3(0.0, 0.25, 0.5))));

              float timegroup = floor(4.0 * dice.x);

              if (beats < i_TENKAI_RGB_DELAY_4FLOOR) {
                // b2sSwing
                float st = 4.0 * beats;
                st = 2.0 * floor(st / 2.0) + step(i_SWING, fract(0.5 * st));

                st = clamp(st, 4.0 * i_TENKAI_HELLO_RGB_DELAY + 10.0, 4.0 * i_TENKAI_RGB_DELAY_4FLOOR - 16.0);
                st += floor(st / 32.0);
                st -= 1.0 + 3.0 * timegroup;
                st = lofi(st, 12.0);
                st += 1.0 + 3.0 * timegroup;
                st -= floor(st / 32.0);

                float i_bst = 0.5 * (floor(st / 2.0) + i_SWING * mod(st, 2.0));
                float t = beats - i_bst;

                col *= vec3(1.0, 0.04, 0.1) * step(0.0, t) * exp(-4.0 * t);
              } else if (beats < i_TENKAI_BREAK) {
                float b = beats;

                b = clamp(b, i_TENKAI_RGB_DELAY_4FLOOR + 3.0, i_TENKAI_FULLHOUSE);
                b -= timegroup;
                b = lofi(b, 4.0);
                b += timegroup;

                float t = beats - b;

                col *= step(0.0, t) * exp(-2.0 * t) * smoothstep(0.0, -4.0, beats - i_TENKAI_BREAK);
              } else {
                float thr = pow(fract(dice.x * 999.0), 0.5);

                col *= smoothstep(0.0, 4.0, beats - i_TENKAI_BREAK - 32.0 * thr) * mix(vec3(1.0), vec3(1.0, 0.05, 0.12), ease(saturate(beats - i_TENKAI_TRANS), 3.0));
              }

              float phase = (
                1.0
                + max(beats - 0.3 * off - timegroup + 0.3 - i_TENKAI_BREAK, 0.0) / 4.0
                + max(beats - 0.3 * off - timegroup + 0.3 - i_TENKAI_FULLHOUSE, 0.0) / 4.0
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
                cp *= 6.0;
                shape *= step(length(fract(cp) - 0.5), 0.3);
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
                // blink rect
                cp = floor(5.0 * cp + 0.5);
                float i_rand = floor(16.0 * phase);
                float i_shape = step(
                  hash3f(vec3(cp, dice.y + i_rand)).x,
                  0.5 - 0.5 * cos(PI * ephase)
                );
                emissive += col * i_shape;
              } else if (dice.z < 8) {
                // char
                float i_rand = floor(30.0 * min(fract(phase), 0.2)) + floor(phase);
                int i_char = int(64.0 * hash3f(dice + i_rand).x);
                float i_d = sddomainchar(14.0 * cp + vec2(4.0, 0.0), i_char);
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
      outColor.xyz += colRem * emissive;

      // if mask is set, break
      if (mask > 0.0) {
        break;
      }

      // the ray missed all of the above, you suck
      if (isect.w >= FAR) {
        float i_intro = 0.5 * smoothstep(0.0, 32.0, beats) * (0.01 + smoothstep(32.0, 31.5, beats));
        outColor.xyz += colRem * i_intro;
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

    if (beats < i_TENKAI_HELLO_RGB_DELAY) {
      float phase = (float(i - 1) + seed.x) / i_SAMPLES_F;
      float diffuse = phase * phase * phase * phase;
      p += (exp(-0.08 * beats) * diffuse + 0.5 * exp(-0.4 * beats) * phase) * cyclic(vec3(4.0 * p, 0.2 * time) + 5.0, 0.5, 1.4).xy;

      float d = 100.0;

      // planefiller
      p = p * 120.0;
      p.x += 1.5 + 0.5 * sddomainspace(47);
      p.x += 1.5 + 0.5 * sddomainspace(43);
      p.x += 1.5 + 0.5 * sddomainspace(32);
      p.x += 1.5 + 0.5 * sddomainspace(45);
      p.x += 1.5 + 0.5 * sddomainspace(36);
      p.x += 1.5 + 0.5 * sddomainspace(37);
      p.x += 1.5 + 0.5 * sddomainspace(40);
      p.x += 1.5 + 0.5 * sddomainspace(43);
      p.x += 1.5 + 0.5 * sddomainspace(43);
      p.x += 1.5 + 0.5 * sddomainspace(36);
      p.x += 1.5 + 0.5 * sddomainspace(49);
      p.x -= 2.0;

      d = min(d, sddomaincharspace(p.xy, 47, 3.0));
      d = min(d, sddomaincharspace(p.xy, 43, 3.0));
      d = min(d, sddomaincharspace(p.xy, 32, 3.0));
      d = min(d, sddomaincharspace(p.xy, 45, 3.0));
      d = min(d, sddomaincharspace(p.xy, 36, 3.0));
      d = min(d, sddomaincharspace(p.xy, 37, 3.0));
      d = min(d, sddomaincharspace(p.xy, 40, 3.0));
      d = min(d, sddomaincharspace(p.xy, 43, 3.0));
      d = min(d, sddomaincharspace(p.xy, 43, 3.0));
      d = min(d, sddomaincharspace(p.xy, 36, 3.0));
      d = min(d, sddomaincharspace(p.xy, 49, 3.0));

      // render
      float shape = smoothstep(2.0 * diffuse, 0.0, d - 0.2);
      vec3 i_col = 3.0 * (0.5 - 0.5 * cos(TAU * saturate(1.5 * phase - vec3(0.0, 0.25, 0.5))));
      outColor.xyz += shape * i_col * smoothstep(-1.0, -4.0, beats - i_TENKAI_HELLO_RGB_DELAY);
    }
  }

  outColor.xyz = mix(
    smoothstep(
      vec3(-0.0, -0.1, -0.2),
      vec3(1.0, 1.1, 1.2),
      sqrt(outColor.xyz / i_SAMPLES_F)
    ),
    max(texture(backBuffer0, uv), 0.0).xyz,
    0.5
  ) * smoothstep(0.0, 4.0, beats) * smoothstep(i_TENKAI_FADEOUT1, i_TENKAI_FADEOUT0, beats);
}
