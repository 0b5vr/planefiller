#version 430

#define saturate(x) clamp(x, 0.0, 1.0)
#define linearstep(a, b, x) min(max(((x) - (a)) / ((b) - (a)), 0.0), 1.0)
#define lofi(i, m) (floor((i) / (m)) * (m))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define inRange(t,a,b) (step(a,t)*(1.-step(b,t)))
#define inRangeB(t,a,b) ((a<=t)&&(t<b))
#define tri(p) (1.-4.*abs(fract(p)-0.5))

const float PI = acos( -1.0 );
const float TAU = PI * 2.0;
const float GOLD = PI * (3.0 - sqrt(5.0));// 2.39996...

const float BPS = 2.25;
const float B2T = 1.0 / BPS;
const float SAMPLES_PER_SEC = 48000.0;
const float SWING = 0.62;

int SAMPLES_PER_BEAT = int(SAMPLES_PER_SEC / BPS);

// https://www.shadertoy.com/view/XlXcW4
vec3 hash3f(vec3 s) {
  uvec3 r = floatBitsToUint(s);
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  return vec3(r) / float(-1u);
}

vec2 cis(float t) {
  return vec2(cos(t), sin(t));
}

mat2 rotate2D( float x ) {
  vec2 v = cis(x);
  return mat2(v.x, v.y, -v.y, v.x);
}

layout(location = 0) uniform int waveOutPosition;
#if defined(EXPORT_EXECUTABLE)
  #pragma work_around_begin:layout(std430,binding=0)buffer ssbo{vec2 %s[];};layout(local_size_x=1)in;
  vec2 waveOutSamples[];
  #pragma work_around_end
#else
  layout(std430, binding = 0) buffer SoundOutput{ vec2 waveOutSamples[]; };
  layout(local_size_x = 1) in;
#endif


float t2sSwing(float t) {
  float st = 4.0 * t / B2T;
  return 2.0 * floor(st / 2.0) + step(SWING, fract(0.5 * st));
}

float s2tSwing(float st) {
  return 0.5 * B2T * (floor(st / 2.0) + SWING * mod(st, 2.0));
}

vec2 shotgun( float t, float spread ) {
  vec2 sum = vec2(0.0);

  for (int i = 0; i ++ < 64;) {
    vec3 dice = hash3f(i + vec3(7, 1, 3));
    sum += vec2(sin(TAU * t * exp2(spread * dice.x))) * rotate2D(TAU * dice.y);
  }

  return sum / 64.0;
}

vec2 ladderLPF(float freq, float cutoff, float reso) {
  float omega = freq / cutoff;
  float omegaSq = omega * omega;

  float a = 4.0 * reso + omegaSq * omegaSq - 6.0 * omegaSq + 1.0;
  float b = 4.0 * omega * (omegaSq - 1.0);

  return vec2(
    1.0 / sqrt(a * a + b * b),
    atan(a, b)
  );
}

vec2 twoPoleHPF(float freq, float cutoff, float reso) {
  float omega = freq / cutoff;
  float omegaSq = omega * omega;

  float a = 2.0 * (1.0 - reso) * omega;
  float b = omegaSq - 1.0;

  return vec2(
    omegaSq / sqrt(a * a + b * b),
    atan(a, b)
  );
}

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 x = normalize(cross(vec3(0, 1, 0), z));
  vec3 y = cross(z, x);
  return mat3(x, y, z);
}

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

void main() {
  const float i_TENKAI_HELLO_RIM = 32.0;
  const float i_TENKAI_HELLO_RAVE = 32.0;
  const float i_TENKAI_HELLO_KICK = 64.0;
  const float i_TENKAI_HELLO_HIHAT = 96.0;
  const float i_TENKAI_HELLO_BASS = 96.0;
  const float i_TENKAI_HELLO_HIHAT_16TH = 128.0;
  const float i_TENAKI_HELLO_CLAP = 128.0;
  const float i_TENKAI_FULLHOUSE = 160.0;
  const float i_TENKAI_TRANS = 192.0;

  int frame = int(gl_GlobalInvocationID.x) + waveOutPosition;
  int tempoutframe = frame;
  frame += int(0) * SAMPLES_PER_BEAT;
  vec4 time = vec4(frame % (SAMPLES_PER_BEAT * ivec4(1, 4, 32, 65536))) / SAMPLES_PER_SEC;
  vec4 beats = time * BPS;

  const bool i_condKickHipass = (
    ((i_TENKAI_HELLO_BASS - 3.0) <= beats.w && beats.w < i_TENKAI_HELLO_BASS)
  );

  const float i_timeCrash = float(frame - SAMPLES_PER_BEAT * int(
    i_TENKAI_TRANS <= beats.w ? i_TENKAI_TRANS :
    i_TENKAI_FULLHOUSE <= beats.w ? i_TENKAI_FULLHOUSE :
    i_TENKAI_HELLO_BASS <= beats.w ? i_TENKAI_HELLO_BASS :
    i_TENKAI_HELLO_KICK <= beats.w ? i_TENKAI_HELLO_KICK :
    -111.0
  )) / SAMPLES_PER_SEC;

  vec2 dest = vec2(0);
  float sidechain = 1.0;

  if (i_TENKAI_HELLO_KICK <= beats.w) { // kick
    float t = time.x;

    float env = smoothstep(0.3, 0.2, t);

    if (i_condKickHipass) { // hi-pass like
      env *= exp(-60.0 * t);
    }

    vec2 wave = vec2(0.0);
    vec2 phase = vec2(44.0 * t);
    phase -= 9.0 * exp(-22 * t);
    phase -= 3.0 * exp(-44 * t);
    phase -= 3.0 * exp(-666 * t);
    wave += sin(TAU * phase);

    dest += 0.5 * env * tanh(1.3 * wave);

    sidechain = smoothstep(0.0, 0.7 * B2T, t) * smoothstep(B2T, 0.99 * B2T, t);
  }

  if (96.0 < beats.w) { // hihat
    float t = mod(time.y, 4.0 * B2T);
    float st = t2sSwing(t);
    st = mix(
      st < 2.0 ? st : lofi(st, 2.0),
      st,
      step(i_TENKAI_HELLO_HIHAT_16TH, beats.w)
    );
    float stt = s2tSwing(st);
    t -= stt;

    float vel = fract(st * 0.38);
    float env = exp(-exp2(6.0 - 1.0 * vel - float(mod(st, 4.0) == 2.0)) * t);
    vec2 wave = shotgun(6000.0 * t, 2.0);
    dest += 0.13 * env * mix(0.2, 1.0, sidechain) * tanh(8.0 * wave);
  }

  if (i_TENKAI_FULLHOUSE <= beats.w) { // open hihat
    float t = mod(time.x - 0.5 * B2T, B2T);

    vec2 sum = vec2(0.0);

    for (int i = 0; i ++ < 8;) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * exp(-5.0 * t) * sin(wave + exp2(13.30 + 0.1 * dice.x) * t + dice2.xy);
      wave = 3.2 * exp(-1.0 * t) * sin(wave + exp2(11.78 + 0.3 * dice.y) * t + dice2.yz);
      wave = 1.0 * exp(-5.0 * t) * sin(wave + exp2(14.92 + 0.2 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.1 * exp(-10.0 * t) * sidechain * tanh(2.0 * sum);
  }

  if (i_TENKAI_HELLO_RIM <= beats.w) { // rim
    float t = time.y;
    float st = t2sSwing(t);
    float stt = s2tSwing(st);
    t -= stt;

    float env = step(0.0, t) * exp(-300.0 * t) * step(0.3, fract(0.38 * st));

    float wave = tanh(4.0 * (
      + tri(t * 400.0 - 0.5 * env)
      + tri(t * 1500.0 - 0.5 * env)
    ));

    dest += 0.2 * env * vec2(wave) * rotate2D(st);
  }

  if (i_TENKAI_FULLHOUSE <= beats.w) { // ride
    float t = mod(time.x, 0.5 * B2T);
    float q = 0.5 * B2T - t;

    float env = exp(-3.0 * t) * smoothstep(0.0, 0.01, q);

    vec2 sum = vec2(0.0);

    for (int i = 0; i ++ < 8;) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 2.9 * env * sin(wave + exp2(13.10 + 0.4 * dice.x) * t + dice2.xy);
      wave = 2.8 * env * sin(wave + exp2(14.97 + 0.4 * dice.y) * t + dice2.yz);
      wave = 1.0 * env * sin(wave + exp2(14.09 + 1.0 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.04 * env * mix(0.3, 1.0, sidechain) * tanh(sum);
  }

  if (i_TENAKI_HELLO_CLAP <= beats.w) { // clap
    float t = mod(time.y - B2T, 2.0 * B2T);

    float env = mix(
      exp(-50.0 * t),
      exp(-400.0 * mod(t, 0.012)),
      exp(-80.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclic(vec3(4.0 * cis(800.0 * t), 840.0 * t), 0.5, 2.0).xy;

    dest += 0.12 * tanh(20.0 * env * wave);
  }

  { // crash
    float t = i_timeCrash;

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(4000.0 * t, 2.5);
    dest += 0.4 * env * mix(0.1, 1.0, sidechain) * tanh(8.0 * wave);
  }

  { // chord stuff
    const int N_CHORD = 8;
    const int CHORD[N_CHORD] = int[](
      0, 7, 10, 12, 15, 17, 19, 22
    );

    float t = mod(time.z, 8.0 * B2T);
    float st = max(1.0, lofi(mod(t2sSwing(t) - 1.0, 32.0), 3.0) + 1.0);
    float stt = s2tSwing(st);
    t = mod(t - stt, 8.0 * B2T);
    float nst = min(st + 3.0, 33.0);
    float nstt = s2tSwing(nst);
    float l = nstt - stt;
    float q = l - t;

    if (beats.w < i_TENKAI_HELLO_RAVE) {
      t = time.z;
      q = i_TENKAI_HELLO_RAVE * B2T - t;
    }

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    float trans = 3.0 * step(beats.w, i_TENKAI_TRANS) + step(i_TENKAI_HELLO_RAVE, beats.w) * step(st, 3.0);

    if (i_TENKAI_HELLO_BASS <= beats.w) { // bass
      float note = 24.0 + trans + float(CHORD[0]);
      float freq = p2f(note);
      float phase = freq * t;
      float wave = tanh(2.0 * sin(TAU * phase));

      dest += 0.5 * sidechain * env * wave;
    }

    if (beats.w < i_TENKAI_HELLO_RAVE) { // longnote
      env *= smoothstep(1.0, i_TENKAI_HELLO_RAVE, beats.w);
    } else if (i_TENKAI_FULLHOUSE < beats.w) { // longer env
      env *= mix(
        smoothstep(1.0 * l, 0.8 * l, t),
        exp(-t),
        0.1
      );
    } else { // shorter env
      env *= mix(
        smoothstep(0.6 * l, 0.4 * l, t),
        exp(-t),
        0.1
      );
    }

    { // choir
      vec2 sum = vec2(0.0);

      for (int i = 0; i ++ < 64;) {
        float fi = float(i);
        vec3 dice = hash3f(i + vec3(5, 4, 2));

        float note = 48.0 + trans + float(CHORD[i % N_CHORD]);
        float freq = p2f(note) * exp(0.012 * tan(2.0 * dice.y - 1.0));
        // float freq = p2f(note) * exp(0.012 * tan(2.0 * dice.x - 1.0));
        float phase = lofi(t * freq, 1.0 / 16.0);

        vec3 c = vec3(0.0);
        vec3 d = vec3(2.0, -3.0, -8.0);
        vec2 wave = 1.0 * cyclic(fract(phase) * d, 0.5, 2.0).xy;

        sum += vec2(wave) * rotate2D(fi);
      }

      dest += 0.04 * mix(0.1, 1.0, sidechain) * env * sum;
    }

    if (i_TENKAI_FULLHOUSE <= beats.w) { // arp
      int iarp = int(16.0 * t / B2T);
      float note = 48.0 + trans + float(CHORD[iarp % N_CHORD]) + 12.0 * float((iarp % 3) / 2);
      float freq = p2f(note);
      float phase = TAU * lofi(t * freq, 1.0 / 16.0);

      vec2 wave = cyclic(vec3(cis(phase), iarp), 0.5, 2.0).xy * rotate2D(time.w);

      dest += 0.2 * sidechain * env * wave;
    }
  }

  waveOutSamples[tempoutframe] = clamp(1.3 * tanh(dest), -1.0, 1.0);
}
