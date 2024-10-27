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
const float SQRT3 = sqrt(3.0);
const float SQRT3_OVER_TWO = SQRT3 / 2.0;

const float i_BPS = 2.25;
const int i_SAMPLES = 20;
const float i_SAMPLES_F = 20.0;
const int i_REFLECTS = 3;

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

// == anim utils ===================================================================================
float ease(float t, float k) {
  float tt = fract(1.0 - t);
  return floor(t) - (k + 1.0) * pow(tt, k) + k * pow(tt, k + 1.0);
}

// == 2d sdfs ======================================================================================
float sdcapsule2(vec2 p, vec2 tail) {
  float i_t = saturate(dot(p, tail) / dot(tail, tail));
  return length(p - i_t * tail);
}

// == text =========================================================================================
vec2 domaingrid(vec2 v) {
  return v + vec2(-0.5, 0.5) * (step(2.0, v) + step(5.0, v)) - vec2(2.5, 3.5);
}

float sddomainseg(vec2 p, vec2 a, vec2 b) {
  a = domaingrid(a);
  b = domaingrid(b);
  return sdcapsule2(p - a, b - a);
}

float sddoamina(vec2 p) {
  float d = sddomainseg(p, vec2(0, 0), vec2(0, 3));
  d = min(d, sddomainseg(p, vec2(0, 3), vec2(2, 6)));
  d = min(d, sddomainseg(p, vec2(2, 6), vec2(4, 6)));
  d = min(d, sddomainseg(p, vec2(4, 6), vec2(6, 3)));
  d = min(d, sddomainseg(p, vec2(6, 3), vec2(6, 0)));
  d = min(d, sddomainseg(p, vec2(0, 2), vec2(6, 2)));
  return d;
}

float sddomainchar(vec2 p, int c) {
  const int VERTICES[] = int[](16,12,11,10,16,15,36,35,05,65,01,61,16,10,56,50,65,56,16,05,04,13,53,62,61,50,10,01,36,30,06,26,24,04,06,42,62,60,40,42,66,65,01,00,63,53,42,41,30,10,01,02,13,33,44,45,36,16,05,04,13,33,42,41,50,60,16,15,36,25,21,30,06,15,11,00,36,32,65,03,05,63,35,31,03,63,11,00,03,43,11,10,66,65,01,00,10,50,61,65,56,16,05,01,10,65,01,16,26,35,30,10,50,05,16,56,65,64,53,13,02,00,60,05,16,56,65,64,53,33,53,62,61,50,10,01,06,04,13,63,66,60,06,66,06,04,54,63,61,50,10,01,56,16,05,01,10,50,61,62,53,03,06,66,65,11,10,56,16,05,04,13,53,62,61,50,10,01,02,13,53,64,65,56,63,13,04,05,16,56,65,61,50,10,01,14,13,11,10,14,13,11,00,65,03,61,04,64,01,61,05,63,01,05,16,56,65,64,53,33,32,31,30,36,33,42,52,63,65,56,16,05,01,10,50,00,03,26,46,63,60,02,62,00,06,56,65,64,53,03,53,62,61,50,00,65,56,16,05,01,10,50,61,06,00,06,56,65,61,50,00,06,66,03,63,06,00,60,06,66,03,53,06,00,65,56,16,05,01,10,50,61,63,43,06,00,03,63,66,60,16,10,66,61,50,10,01,02,06,00,66,65,43,03,43,61,60,06,00,60,00,06,33,32,33,66,60,00,06,60,66,10,50,61,65,56,16,05,01,10,00,06,56,65,64,53,03,10,50,61,65,56,16,05,01,10,42,61,00,06,56,65,64,53,03,53,62,60,65,56,16,05,04,13,53,62,61,50,10,01,06,66,36,30,06,01,10,50,61,66,06,03,20,40,63,66,06,00,33,34,33,60,66,06,05,61,60,66,65,01,00,06,05,33,66,65,33,30,06,66,65,01,00,60,36,26,20,30,06,05,61,60,06,16,10,00,14,36,54,00,60,06,15);
  const int SEGMENTS[] = int[](0,2,4,6,8,10,12,14,16,28,30,35,40,44,66,68,72,76,78,80,82,84,86,88,90,92,96,105,107,111,113,123,130,136,140,142,144,152,162,167,184,195,197,199,201,203,206,208,210,213,221,223,235,241,243,250,255,263,265,271,273,275,278,280,282,284,294,296,298,300,302,308,310,314,317,320,324,327,331,340,347,356,358,365,368,380,382,384,390,396,400,403,407,411,414,418,424,428,432,436,439,441,443);
  const int CHARS[] = int[](0,2,4,8,10,13,14,15,16,17,20,22,23,24,25,26,28,30,31,33,35,37,38,39,40,41,43,45,46,48,49,51,52,54,56,57,59,62,65,66,69,70,71,74,75,77,78,79,80,82,84,85,87,88,89,91,93,95,96,97,98,99,100,101,102);

  float d = 1.0;

  if (abs(p.x) > 3.0 || abs(p.y) > 4.0) {
    return 1.0;
  }

  int i_char = CHARS[c];
  for (int i = CHARS[c]; i < CHARS[c + 1]; i ++) {
    int seg0 = SEGMENTS[i];
    int seg1 = SEGMENTS[i + 1] - 1;

    for (int i = seg0; i < seg1; i ++) {
      int v0 = VERTICES[i];
      int v1 = VERTICES[i + 1];

      d = min(d, sddomainseg(p, vec2(v0 / 10, v0 % 10), vec2(v1 / 10, v1 % 10)));
    }
  }

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
  outColor *= 0.0;

  vec2 uv = gl_FragCoord.xy / resolution.xy;

  float time = globalTime;
  vec3 seed = hash3f(vec3(uv, time));
  float beat = time * i_BPS;
  float beatpulse = 0.2 + 0.8 * pow(0.5 - 0.5 * cos(TAU * ease(beat, 7.0)), 0.3);
  float beatpulse2 = exp(-5.0 * fract(beat));

  for (int i = 0; i ++ < i_SAMPLES;) {
    vec2 p = (uv - 0.5) + (seed = hash3f(seed)).xy / resolution.y;
    p.x *= resolution.x / resolution.y;

    vec3 colRem = vec3(0.4, 0.2, 1.0);

    mat3 cb = orthBas(colRem);
    vec3 ro = 10.0 * cb[2];
    vec3 rd = cb * normalize(vec3(p, -10.0));
    vec3 fp = ro + rd * 9.0;
    ro += cb * vec3(0.01 * uniformSphere((seed = hash3f(seed)).xy).xy, 0.0);
    ro += rd * mix(5.0, 6.0, seed.x);
    rd = normalize(fp - ro);
    ro.z -= 0.4 * time;

    colRem *= (1.0 - 0.5 * length(p)) / colRem;

    const float i_PLANE_INTERVAL = 0.5;

    for (int i = 0; i ++ < i_REFLECTS;) {
      vec3 emissive = vec3(0.0);
      float roughness = 0.3;
      float mask = 0.0;

      // floor
      vec4 isect = vec4(0.0, 1.0, 0.0, -ro.y / rd.y);
      if (isect.w < 0.0) {
        isect = vec4(FAR);
      }

      // floor greebles quadtree shit
      const float i_GAP = 0.01;
      const float i_GREEBLES_HEIGHT = 0.01;
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
            if (dice.x < 0.1) {
              break;
            }
            size.xz /= 1.0 + vec2(step(0.7, dice.y), step(dice.y, 0.4));
          }

          cell = lofi(gro, 2.0 * size) + size;
          cell.y = 0.0;
          dice = hash3f(cell);
        }

        vec3 i_size = size - vec2((1.0 - beatpulse) * dice.z * i_GREEBLES_HEIGHT, i_GAP).yxy;
        vec4 isect2 = isectBox(ro - cell, rd, i_size);
        if (isect2.w < isect.w) {
          isect = isect2;
          dice = hash3f(dice);
          roughness = exp(-1.0 - dice.y);
          break;
        }

        // forward to the next cell
        grl += isectIBox(gro - cell, rd, size).w + 0.01;
      }

      // plane array
      float side = sign(rd.z);
      float planez = (floor(ro.z / i_PLANE_INTERVAL) + 0.5 * (1.0 + side)) * i_PLANE_INTERVAL;

      for (int i = 0; i ++ < 32;) {
        vec4 isect2 = vec4(0.0, 0.0, -side, abs((ro.z - planez) / (rd.z)));

        // if the plane is already further than existing isect, break
        if (isect.w < isect2.w) {
          break;
        }

        vec3 rp = ro + rd * isect2.w;
        rp.y -= i_GREEBLES_HEIGHT;

        vec3 id = vec3(planez + vec3(1, 2, 3));
        vec3 dice = hash3f(id);

        float kind = fract(1.52 * planez);
        if (kind < 0.1) {
          // rainbow bar
          if (abs(rp.y - 0.02) < 0.01) {
            mask = 1.0;
            float i_phase = TAU * dice.z + rp.x;
            vec3 i_rainbow = 1.0 + cos(i_phase + vec3(0, 2, 4));
            emissive += 10.0 * mask * i_rainbow * beatpulse;
          }
        } else if (kind < 0.2) {
          // large pillar
          mask = step(abs(abs(rp.x) - 0.5), 0.05);
          emissive += vec3(5.0, 8.0, 10.0) * mask * beatpulse;
        } else if (kind < 0.3) {
          // rave laser
          rp.y += 0.01;
          float t = dice.y + floor(beat);
          float d = min(
            max(abs(mod((rp.xy * rotate2D(t)).x, 0.04) - 0.02), 0.0),
            max(abs(mod((rp.xy * rotate2D(-t)).x, 0.04) - 0.02), 0.0)
          );
          emissive += smoothstep(2.0, 0.0, abs(rp.x)) * exp(-4.0 * rp.y) * beatpulse2 * step(d, 0.001) * vec3(0.1, 10.0, 2.0);
        } else if (kind < 0.5) {
          // huge stuff
          dice = hash3f(dice + floor(beat));
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
        } else if (abs(rp.x) < 1.0) {
          // rgb delay shit
          float size = 0.25;
          dice = hash3f(vec3(floor(rp.xy / size), dice.z));
          size /= 1.0 + step(0.5, dice.z);
          dice = hash3f(vec3(floor(rp.xy / size), dice.z));
          size /= 1.0 + step(0.5, dice.z);
          dice = hash3f(vec3(floor(rp.xy / size), dice.z));
          vec2 cp = rp.xy / size;

          if (abs(cp.y - 0.5) < 0.5 && dice.y < 0.5) {
            cp = (fract(cp.xy) - 0.5) * size / (size - 0.01);

            if (abs(cp.x) < 0.5 && abs(cp.y) < 0.5) {
              float off = (seed = hash3f(seed)).y;
              vec3 col = 6.0 * (0.5 - 0.5 * cos(TAU * saturate(1.5 * off - vec3(0.0, 0.25, 0.5))));

              float timegroup = floor(4.0 * dice.y);
              float phase = beat / 2.0 - 0.2 * off - timegroup / 2.0;
              float ephase = ease(phase, 6.0);
              float ephase0 = min(mod(ephase, 2.0), 1.0);
              float ephase1 = max(mod(ephase, 2.0) - 1.0, 0.0);

              if (dice.z < 0.2) {
                // circle
                emissive += col * step(0.5 * ephase0 - 0.2, length(cp)) * step(length(cp), 0.5 * ephase0) * step(1.1 * ephase1, fract(atan(cp.y, cp.x) / TAU - ephase1 - 2.0 * TAU * dice.y));
              } else if (dice.z < 0.4) {
                // dot matrix
                float shape = step(abs(cp.y), 0.5) * step(abs(cp.x), 0.5) * step(length(fract(8.0 * cp) - 0.5), 0.3);
                cp = floor(8.0 * cp);
                float i_rand = floor(12.0 * min(fract(phase), 0.5));
                emissive += col * shape * step(
                  hash3f(vec3(cp, dice.y + i_rand)).x,
                  0.3 - 0.3 * cos(PI * ephase)
                );
              } else if (dice.z < 0.6) {
                // hex
                cp.y += 0.05;
                cp *= rotate2D(TAU * lofi(dice.y - ephase, 1.0 / 6.0));
                float cell = floor(atan(cp.x, cp.y) / TAU * 6.0 + 0.5);
                cp *= rotate2D(cell / 6.0 * TAU);
                float i_shape = (
                  step(0.02, dot(abs(cp), vec2(-SQRT3_OVER_TWO, 0.5)))
                  * step(0.24, cp.y)
                  * step(cp.y, 0.44)
                ) * step(mod(cell, 3.0), 1.0 - 1.1 * cos(PI * ephase));
                emissive += col * i_shape;
              } else if (dice.z < 0.8) {
                // blink rect
                cp = floor(5.0 * cp + 0.5);
                float i_rand = floor(16.0 * phase);
                float i_shape = step(
                  hash3f(vec3(cp, dice.y + i_rand)).x,
                  0.5 - 0.5 * cos(PI * ephase)
                );
                emissive += col * i_shape;
              } else if (dice.z < 0.9) {
                // char
                float i_rand = floor(30.0 * min(fract(phase), 0.2)) + floor(phase);
                int i_char = int(64.0 * hash3f(dice + i_rand).x);
                float i_d = sddomainchar(10.0 * cp, i_char);
                emissive += col * step(i_d, 0.2);
              } else {
                // arrow
                cp /= ephase0;

                float blink = floor(min(8.0 * ephase1, 3.0));

                vec2 cpt = vec2(
                  abs(cp.x),
                  0.8 - fract(cp.y + 0.5 - 2.0 * ephase0)
                );
                float d = min(
                  sdcapsule2(cpt, vec2(0.0, 0.6)),
                  sdcapsule2(cpt, vec2(0.3, 0.3))
                ) - 0.07;

                if (blink < 2.0) {
                  cpt = cp;
                  cpt -= clamp(cpt, -0.4, 0.4);
                  d = max(-d, length(cpt) - 0.08);
                }

                float shape = mix(
                  mix(
                    step(d, 0.0),
                    step(abs(d), 0.01),
                    saturate(blink)
                  ),
                  mix(
                    step(d, 0.0),
                    0.0,
                    saturate(blink - 2.0)
                  ),
                  saturate(blink - 1.0)
                );
                emissive += col * shape;
              }
            }
          }
        }

        // if the mask test misses, traverse the next plane
        if (mask == 0.0) {
          planez += i_PLANE_INTERVAL * side;
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
  }

  outColor.xyz = mix(
    smoothstep(
      vec3(0.0, -0.1, -0.2),
      vec3(1.0, 1.1, 1.2),
      sqrt(outColor.xyz / i_SAMPLES_F)
    ),
    max(texture(backBuffer0, uv), 0.0).xyz,
    0.0
  );
}
