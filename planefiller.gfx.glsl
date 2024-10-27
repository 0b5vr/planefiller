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
#define linearstep(a, b, x) min(max(((x) - (a)) / ((b) - (a)), 0.0), 1.0)
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
    ro += rd * mix(5.0, 6.0, seed.x);
    vec3 fp = ro + rd * 4.0;
    ro += cb * vec3(0.01 * uniformSphere((seed = hash3f(seed)).xy).xy, 0.0);
    rd = normalize(fp - ro);
    ro.z -= 0.4 * time;

    colRem *= (1.0 - 0.5 * length(p)) / colRem;

    const float i_PLANE_INTERVAL = 0.5;

    for (int i = 0; i ++ < i_REFLECTS;) {
      vec3 baseColor = vec3(0.3);
      vec3 emissive = vec3(0.0);
      float roughness = 0.3;
      float metallic = 1.0;

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
          metallic = step(0.5, dice.x);
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

        float mask = 0.0;
        float kind = fract(0.62 * planez);
        if (kind < 0.1) {
          // rainbow bar
          if (abs(rp.y - 0.01) < 0.01) {
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
            // slash
            rp.y -= 0.25;
            mask = max(abs(rp.x) - 0.25, abs(rp.y) - 0.25);
            mask = max(
              max(
                step(abs(rp.x + rp.y), 0.002),
                step(abs(rp.x - rp.y), 0.002)
              ) * step(mask, 0.0),
              step(abs(mask), 0.001)
            );
          } else if (dice.x < 0.75) {
            // dashed box
            dice.yz = exp2(-4.0 * dice.yz);
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
              float i_off = (seed = hash3f(seed)).y;
              vec3 col = 6.0 * (0.5 - 0.5 * cos(TAU * saturate(1.5 * i_off - vec3(0.0, 0.25, 0.5))));

              float phase = beat / 2.0 - 0.2 * i_off - floor(4.0 * dice.y) / 2.0;
              phase = ease(phase, 4.0);
              float phase0 = min(mod(phase, 2.0), 1.0);
              float phase1 = max(mod(phase, 2.0) - 1.0, 0.0);

              if (dice.z < 0.2) {
                // circle
                emissive += col * step(0.5 * phase0 - 0.2, length(cp)) * step(length(cp), 0.5 * phase0) * step(1.1 * phase1, fract(atan(cp.y, cp.x) / TAU - phase1 - 2.0 * TAU * dice.y));
              } else if (dice.z < 0.4) {
                // dot matrix
                float shape = step(abs(cp.y), 0.5) * step(abs(cp.x), 0.5) * step(length(fract(8.0 * cp) - 0.5), 0.3);
                cp = floor(8.0 * cp);
                emissive += col * shape * step(
                  hash3f(vec3(cp, dice.y + floor(5.0 * phase - 0.1))).x,
                  0.3 + 0.3 * cos(PI * phase)
                );
              } else if (dice.z < 0.6) {
                // hex
                cp.y += 0.05;
                cp *= rotate2D(TAU * lofi(dice.y - phase, 1.0 / 6.0));
                float cell = floor(atan(cp.x, cp.y) / TAU * 6.0 + 0.5);
                cp *= rotate2D(cell / 6.0 * TAU);
                float i_shape = (
                  step(0.02, dot(abs(cp), vec2(-SQRT3_OVER_TWO, 0.5)))
                  * step(0.24, cp.y)
                  * step(cp.y, 0.44)
                ) * step(mod(cell, 3.0), 1.0 + 1.1 * cos(PI * phase));
                emissive += col * i_shape;
              } else if (dice.z < 0.8) {
                // dot matrix
                float shape = step(abs(cp.y), 0.5) * step(abs(cp.x), 0.5) * step(length(fract(8.0 * cp) - 0.5), 0.4);
                cp = floor(5.0 * cp + 0.5);
                emissive += col * step(
                  hash3f(vec3(cp, dice.y + floor(5.0 * phase - 0.1))).x,
                  0.5 + 0.5 * cos(PI * phase)
                );
              } else {
                // arrow
                cp /= phase0;

                float blink = floor(min(8.0 * phase1, 3.0));

                vec2 cpt = cp;
                cpt.y = fract(cpt.y + 0.5 - 2.0 * phase0) - 0.5;
                cpt.y -= clamp(cpt.y, -0.3, 0.3);
                float d = length(cpt) - 0.07;
                cpt = vec2(abs(cp.x), cp.y - 0.3);
                cpt.y = fract(cpt.y + 0.5 - 2.0 * phase0) - 0.5;
                cpt *= rotate2D(-PI / 4.0);
                cpt.y -= clamp(cpt.y, -0.4, 0.0);
                d = min(d, length(cpt) - 0.07);

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
        baseColor = vec3(0.0);
        roughness = 1.0;
        metallic = 0.0;
        break;
      }

      // emissive
      outColor.xyz += colRem * emissive;

      // the ray missed all of the above, you suck
      if (isect.w >= FAR) {
        break;
      }

      // now we have a hit

      // early break if baseColor is zero
      if (dot(baseColor, baseColor) < 0.0000001) {
        break;
      }

      // set materials
      ro += isect.w * rd + isect.xyz * 0.001;
      float sqRoughness = roughness * roughness;
      float sqSqRoughness = sqRoughness * sqRoughness;
      float halfSqRoughness = 0.5 * sqRoughness;

      // shading
      {
        float NdotV = dot(isect.xyz, -rd);
        float Fn = mix(0.04, 1.0, pow(1.0 - NdotV, 5.0));
        float spec = max(
          step((seed = hash3f(seed)).x, Fn), // non metallic, fresnel
          metallic // metallic
        );

        // sample ggx or lambert
        seed.y = sqrt((1.0 - seed.y) / (1.0 - spec * (1.0 - sqSqRoughness) * seed.y));
        vec3 woOrH = orthBas(isect.xyz) * vec3(
          sqrt(1.0 - seed.y * seed.y) * sin(TAU * seed.z + vec2(0.0, TAU / 4.0)),
          seed.y
        );

        if (spec > 0.0) {
          // specular
          // note: woOrH is H right now
          vec3 i_H = woOrH;
          vec3 i_wo = reflect(rd, i_H);
          if (dot(i_wo, isect.xyz) < 0.0) {
            break;
          }

          // vector math
          float NdotL = dot(isect.xyz, i_wo);
          float i_VdotH = dot(-rd, i_H);
          float i_NdotH = dot(isect.xyz, i_H);

          // fresnel
          vec3 i_F0 = mix(vec3(0.04), baseColor, metallic);
          vec3 i_Fh = mix(i_F0, vec3(1.0), pow(1.0 - i_VdotH, 5.0));

          // brdf
          // colRem *= Fh / Fn * G * VdotH / ( NdotH * NdotV );
          colRem *= clamp(
            i_Fh / mix(Fn, 1.0, metallic)
              / (NdotV * (1.0 - halfSqRoughness) + halfSqRoughness) // G1V / NdotV
              * NdotL / (NdotL * (1.0 - halfSqRoughness) + halfSqRoughness) // G1L
              * i_VdotH / i_NdotH,
            0.0,
            2.0
          );

          // wo is finally wo
          woOrH = i_wo;
        } else {
          // diffuse
          // note: woOrH is wo right now
          if (dot(woOrH, isect.xyz) < 0.0) {
            break;
          }

          // calc H
          // vector math
          vec3 i_H = normalize(-rd + woOrH);
          float i_VdotH = dot(-rd, i_H);

          // fresnel
          float i_Fh = mix(0.04, 1.0, pow(1.0 - i_VdotH, 5.0));

          // brdf
          colRem *= clamp((1.0 - i_Fh) / (1.0 - Fn) * baseColor, 0.0, 2.0);
        }

        // prepare the rd for the next ray
        rd = woOrH;
      }

      if (dot(colRem, colRem) < 0.01) {
        // break;
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
